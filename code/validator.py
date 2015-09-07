  # Copyright 2015 Antonio Massaro

  #  Licensed under the Apache License, Version 2.0 (the "License");
  #  you may not use this file except in compliance with the License.
  #  You may obtain a copy of the License at

  #      http://www.apache.org/licenses/LICENSE-2.0

# performs a leave-one-out cross-validation of the model(s) developed and returns the confusion matrix
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
n_est=20
n_feat=1

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


def validator(path_to_features,path_to_labels):
	clf=RandomForestClassifier(n_estimators=n_est, max_features=n_feat,max_depth=9, n_jobs=-1)
	labels=load_labels(path_to_labels)
	X=load_features(path_to_features)
	print len(X),len(labels)
	nclasses=len(set(labels))
	Conf=np.zeros((nclasses,nclasses))
	for i in range(len(X)-1):
		train=np.array(list(range(i,i+1)))
		test=np.array(list(range(0,i))+list(range(i+1,len(X))))
		XX = X[train]
		y=labels[train] 
		clf = clf.fit(XX, y)
		true=np.array([int(labels[t]) for t in test])
		pred=np.array([int(clf.predict(X[t])[0]) for t in test])
		Conf[true,pred]+=1 
		if i%10==0:
			print i
	return Conf