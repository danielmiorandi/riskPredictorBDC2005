  # Copyright 2015 Antonio Massaro

  #  Licensed under the Apache License, Version 2.0 (the "License");
  #  you may not use this file except in compliance with the License.
  #  You may obtain a copy of the License at

  #      http://www.apache.org/licenses/LICENSE-2.0

# This file reads raw data from the data/ folder and creates a corresponding features matrix in the features/ folder


from datetime import datetime
from datetime import timedelta
import numpy as np
import os
from itertools import combinations as combo

def percentilize(X):
    X_s=np.sort(X)
    a=[np.where(x==X_s)[0][-1]/float(len(X_s)) for x in X]
    return np.array(a)

def hourly_aggregation(X,h):
	Xh=np.zeros((X.shape[0],X.shape[1]/h))
	for i in range(X.shape[0]):
		x=X[i]
		j=0
		k=0
		while j<len(x)-h:
			Xh[i,k]=sum(x[j:j+h])
			k+=1
			j+=h
	return Xh

def normalize(X):
	Xn=np.zeros(X.shape)
	for i in range(len(X)):
		Xn[i,:]=(X[i,:]-min(X[i,:]))/(max(X[i,:])-min(X[i,:]))
	return Xn

def getfeat():
	start=datetime(2015,3,1,0)
	end=datetime(2015,5,1,0)
	timestep=3600
	n_timeslots=int((end-start).total_seconds()/timestep+1)

	accidents=np.zeros(n_timeslots)
	inf=open('../data/accidents_by_cell_1hour_day','r')
	for line in inf:
	    line=line.split(',')
	    line[-1]=line[-1][:-1]
	    if line[-1]!='accidents':
	        t=datetime.strptime(line[1],'%Y-%m-%d %H:%M:%S')
	        v=float(line[4])
	        accidents[int(((t-start).total_seconds())/timestep)]+=v


	telco=np.zeros((4,n_timeslots))
	files=['../data/callIn','../data/callOut','../data/smsIn','../data/smsOut']
	for i in range(len(files)):
	    f=open(files[i],'r')
	    for line in f:
	        line=line.split(',')
	        t=datetime.strptime(line[0],'%Y-%m-%d %H:%M:%S')
	        telco[i,int(((t-start).total_seconds())/timestep)]=float(line[1])


	traffic=np.zeros(n_timeslots)
	out=open('../data/total_fl_car_traffic','r')
	for line in out:
	    line=line.split(',')
	    t=datetime.strptime(line[0],'%Y-%m-%d %H:%M:%S')
	    traffic[int(((t-start).total_seconds())/timestep)]=float(line[1])
	    
	traffic_unique=np.zeros(n_timeslots)
	out=open('../data/unique_floating_car','r')
	for line in out:
	    line=line.split(',')
	    t=datetime.strptime(line[0],'%Y-%m-%d %H:%M:%S')
	    traffic_unique[int(((t-start).total_seconds())/timestep)]=float(line[1])
	
	traffic_unique_10=np.zeros(n_timeslots)
	out=open('../data/unique_floating_car_10','r')
	for line in out:
	    line=line.split(',')
	    t=datetime.strptime(line[0],'%Y-%m-%d %H:%M:%S')
	    traffic_unique_10[int(((t-start).total_seconds())/timestep)]=float(line[1])
	
	traffic_unique_20=np.zeros(n_timeslots)
	out=open('../data/unique_floating_car_20','r')
	for line in out:
	    line=line.split(',')
	    t=datetime.strptime(line[0],'%Y-%m-%d %H:%M:%S')
	    traffic_unique_20[int(((t-start).total_seconds())/timestep)]=float(line[1])
	
	traffic_unique_40=np.zeros(n_timeslots)
	out=open('../data/unique_floating_car_40','r')
	for line in out:
	    line=line.split(',')
	    t=datetime.strptime(line[0],'%Y-%m-%d %H:%M:%S')
	    traffic_unique_40[int(((t-start).total_seconds())/timestep)]=float(line[1])
	
	traffic_unique_50=np.zeros(n_timeslots)
	out=open('../data/unique_floating_car_50','r')
	for line in out:
	    line=line.split(',')
	    t=datetime.strptime(line[0],'%Y-%m-%d %H:%M:%S')
	    traffic_unique_50[int(((t-start).total_seconds())/timestep)]=float(line[1])
	
	weather=np.zeros(n_timeslots)-1
	out=open('../data/weather','r')
	for line in out:
	    if line[0]!='t':
	        line=line.split(',')
	        t=datetime.strptime(line[0],'%Y-%m-%d %H:%M:%S')
	        weather[int(((t-start).total_seconds())/timestep)]=int(line[1])

	w=list(set(weather))
	Weather=np.array([w.index(ww) for ww in weather])
	D=[]
	i=0
	while i <len(Weather):
	    if Weather[i]==11:
	        s=i
	        c=0
	        while Weather[i+c]==11:
	            c+=1
	        D.append([s,s+c])
	        i=i+c
	    else:
	        i+=1
	for x in D:
	    if x[1]-x[0]<5:
	        for i in range(x[0],x[1]):
	            if i<(x[1]-x[0])/2 and x[0]>0:
	                Weather[i]=Weather[[x[0]-1]]
	            else:
	                Weather[i]=Weather[[x[1]]]
	k=w.index(-1)

	Weather=np.array([Weather[i] if Weather[i]!=k else Weather[i-1] for i in range(len(Weather))])
	Weather=np.array([Weather[i] if Weather[i]!=k else Weather[i+1] for i in range(len(Weather)-1)])


	X=np.zeros((len(telco[0]),12))
	for i in range(4):
	    for j in range(len(telco[0])):
	        X[j,i]=telco[i,j]
	for j in range(len(traffic)):
	    X[j,4]=traffic[j]
	    X[j,5]=traffic_unique[j]
	    X[j,6]=traffic_unique_10[j]
	    X[j,7]=traffic_unique_20[j]
	    X[j,8]=traffic_unique_40[j]
	    X[j,9]=traffic_unique_50[j]

	twtt=[]
	tw=open('../data/twitterVolume.csv','r')
	for l in tw:
		twtt.append(int(l.split(',')[1]))
	twtt=np.array(twtt)
	for j in range(len(twtt)):
		X[j,10]=twtt[j]

	for j in range(len(Weather)):
	    X[j,11]=Weather[j]
     
	names=['callin','callout','smsin','smsout','traffic','traffic_unique','traffic_unique_10','traffic_unique_20','traffic_unique_40','traffic_unique_50','twitter','weather']
	return [X,names]




def featuresExtractor(h):
	start=datetime(2015,3,1,0)
	Q=getfeat()
	X=Q[0][:,:-1].T
	X=hourly_aggregation(X,h)
	names=Q[1][:-1]
	couples=[list(y) for y in combo(list(range(len(X))), 2)]
	triples=[list(y) for y in combo(list(range(len(X))), 3)]
	Xn=normalize(X)
	for c in couples:
		X=np.vstack((X,np.sum(Xn[c],0),np.diff(Xn[c].T)[:,0]))
		names.append(str(c)+'+')
		names.append(str(c)+'-')
	for t in triples:
		X=np.vstack((X,np.sum(Xn[t],0)))
		names.append(str(t))
	n=len(X)
	for i in range(n):
	    f=X[i]
	    f_p=percentilize(f)
	    f_p_100=np.round(f_p*100)
	    f_p_20=np.round(f_p*20)
	    f_p_10=np.round(f_p*10)
	    f_p_5=np.round(f_p*5)
	    X=np.vstack([X,f_p,f_p_100,f_p_20,f_p_10,f_p_5])
	    names+=[names[i]+'_q',names[i]+'_q100',names[i]+'_q20',names[i]+'_q10',names[i]+'_q5']

	weather=Q[0][:,-1]
	weather_h=np.array([weather[i] for i in range(len(weather)) if i%h==0])
	names.append('weather')
	X=np.vstack((X,weather_h[:-1]))
	
	date=datetime(2015,3,1,0)
	timestep=3600
	H=np.array([(start+i*timedelta(0,h*timestep)).hour for i in range(len(X[0]))])
	names.append('hour')
	X=np.vstack((X,H))
	
	outfile=open('../features/features_matrix','w')
	outfile.write('datetime,')
	for n in names:
		outfile.write(str(n)+',')
	outfile.write('\n')
	X=X.T
	for x in X:
		outfile.write(str(date)+',')
		for f in x:
			outfile.write(str(f)+',')
		outfile.write('\n')
		date+=timedelta(0,h*timestep)
	return
	#return [X,names]