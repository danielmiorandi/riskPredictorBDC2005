  # Copyright 2015 Antonio Massaro

  #  Licensed under the Apache License, Version 2.0 (the "License");
  #  you may not use this file except in compliance with the License.
  #  You may obtain a copy of the License at

  #      http://www.apache.org/licenses/LICENSE-2.0

  # starting from the raw data in data/ creates labels for training the model. Results are stored in features/

from datetime import datetime
from datetime import timedelta
import numpy as np
import os  
partition=[[0,1,2,3,4],[5,6,7,8,9]]
time_frame=3

def part(Y):
    
    Y_q=np.zeros(len(Y))
    i=0
    for y in Y:
        k=0
        score=-1
        for p in partition:
            if y in p:
                score=k
                break
            k+=1
        if score==-1:
            score=len(partition)
        Y_q[i]=score
        i+=1
    return Y_q


def labeller():
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
	accidents_h=np.zeros(len(accidents)/time_frame)
	k=0
	j=0
	while j<len(accidents)-time_frame:
		accidents_h[k]=sum(accidents[j:j+time_frame])
		k+=1
		j+=time_frame
	labels=part(accidents_h)
	outfile=open('../features/labels','w')
	outfile.write('datetime,labels\n')
	date=start
	for l in labels:
		outfile.write(str(date)+','+str(l)+'\n')
		date+=timedelta(0,time_frame*timestep)
	outfile.close()
	return 

labeller()