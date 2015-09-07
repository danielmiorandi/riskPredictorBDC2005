# Copyright 2015 Antonio Massaro

  #  Licensed under the Apache License, Version 2.0 (the "License");
  #  you may not use this file except in compliance with the License.
  #  You may obtain a copy of the License at

  #      http://www.apache.org/licenses/LICENSE-2.0

  #This file takes relevant data streams and outputs the predicted risk level (H,M,L). As for the time being historical data from data/ is used, it is sufficient to indicate a date/time/position and the relevant data will be fetchted by the script.
from sklearn.externals import joblib
from datetime import datetime
import numpy as np

def predictor(path_to_classifier,path_to_features,time):
	clf=joblib.load('../models/trained_random_forest.pkl')
	date=datetime.strptime(time,'%Y-%m-%d %H:%M:%S')
	assert(date.day in list(range(1,32)))
	assert(date.month in [3,4])
	assert(date.year==2015)
	f=open(path_to_features,'r')
	k=0
	for line in f:
		if k>0:
			x=line.split(',')
			d=datetime.strptime(x[0],'%Y-%m-%d %H:%M:%S')
			if abs((d-date).total_seconds())<3600*3:
				chosen=x	
				break
		k+=1
	X=np.array([float(c) for c in chosen[1:-1]])
	y=clf.predict(X)
	return y

