  # Copyright 2015 Antonio Massaro

  #  Licensed under the Apache License, Version 2.0 (the "License");
  #  you may not use this file except in compliance with the License.
  #  You may obtain a copy of the License at

  #      http://www.apache.org/licenses/LICENSE-2.0
# trains the classifiers (200 20-features random forests) based on labels and features. The resulting model is saved in models/

import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import numpy as np

n_est=200
n_feat=20

def load_features(path_to_features_matrix):
	f=open(path_to_features_matrix,'r')
	n_samples=-1
	n_features=0
	for line in f:
		n_samples+=1
	f.seek(0)
	for line in f:
		line=line.split(',')[1:-1]
		n_features=len(line)
	f.seek(0)
	X=np.zeros((n_samples,n_features))
	i=-1
	for line in f:
		if i>=0:
			line=line.split(',')[1:-1]
			for j in range(len(line)):
				X[i,j]=float(line[j])
		i+=1
	return X 


def load_labels(path_to_labels):
	f=open(path_to_labels,'r')
	n_samples=-1
	for line in f:
		n_samples+=1
	f.seek(0)
	Y=np.zeros(n_samples)
	i=-1
	for line in f:
		if i>=0:
			line=line.split(',')
			Y[i]=float(line[1])
		i+=1
	return Y

def train(path_to_features_matrix,path_to_labels):
	clf=RandomForestClassifier(n_estimators=n_est, max_features=n_feat,max_depth=9, n_jobs=-1)
	X=load_features(path_to_features_matrix)
	Y=load_labels(path_to_labels)
	clf.fit(X,Y)
	joblib.dump(clf, '../models/trained_random_forest.pkl')
	return clf





