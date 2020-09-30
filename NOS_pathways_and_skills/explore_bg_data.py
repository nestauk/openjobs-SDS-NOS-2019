#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:54:59 2019

@author: stefgarasto

Description:
This file is used to load the job adverts dataset for all years and print out
some information about missing data and value counts. This information includes
the size of the subset of jobs that is likely usable by the prediction algorithm.

The value counts are pickled and then plotted by another script: 
    "bg_plot_counts_histograms", when WHOLEDATA = True

This is only to get information about the WHOLE dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from utils_bg import load_all_bg_data
import pickle

from utils_bg import *

#%% load the data for all years
filename= '/Users/stefgarasto/Local-Data/Burning-glass/Job_ads_2012_2018/{}_reliable_soc_ads.csv'
all_years = ['2012','2013','2014','2015','2016','2017','2018']

t0 = time.time()
for ix,year in enumerate(all_years):
    if ix == 0:
        bgdata = pd.read_csv(filename.format(year))
    else:
        bgdata = pd.concat((bgdata, pd.read_csv(filename.format(year))))
        print(len(bgdata))
print('Time in minutes: {:.4f}'.format((time.time()- t0)/60))


#%%
allcols = bgdata.columns
# only keep columns of interest
coi = ['Unnamed: 0', 'BGTJobId', 'JobDate', 'SOC', 'SOCName', 'Employer',
       'Region_Nation', 'MinDegree', 'MaxDegree', 'MinEdu', 'MaxEdu',
       'MinNQF', 'MaxNQF', 'MinExp', 'MaxExp', 'MinSalary', 'MaxSalary',
       'PayFrequency', 'SalaryType', 'InternshipFlag', 'ApprenticeshipFlag',
       'WorkFromHomeFlag', 'title_processed', 'converted_skills', '3d_SOC',
       'new_soc', 'soc_score', 'clusters', 'trans']
bgdata = bgdata[coi]
allcols = bgdata.columns

#%%
# show percentage of missing values
nullvalues= bgdata.isnull().sum()

#%%
'''
# Print out some interesting information about the data and save
# some of the value counts to be plotted later
'''
Ntot = len(bgdata)
info_about_counts = {}
with open(os.path.join(saveoutput,'bgdata_initial_notes.txt'),'w') as f:
    print('There is a total of {} job adverts in this dataset \n\n\n'.format(Ntot), 
          file = f)
    print('Percentage of missing values in data from 2012 to 2018 \n', file= f)
    print(nullvalues/Ntot*100, file = f)
    print('\n', file = f)
    print(nullvalues/Ntot*100)
    for outputcol in ['MinExp','MaxExp','MinEdu','MaxEdu','new_soc','soc_score']:
        print('Percentages for {}'.format(outputcol), file =f)
        N = Ntot - nullvalues[outputcol]
        #print(outputcol, file =f )
        tmp = bgdata[outputcol].value_counts()
        info_about_counts['Percentages for {}'.format(outputcol)] = tmp/N*100
        info_about_counts['Counts for {}'.format(outputcol)] = tmp
        for ix,t in enumerate(tmp.index):
            print('{} \t \t {:.6f}'.format(t , tmp.loc[t]/N*100), file= f)
        print('\n', file= f)
        print('Counts for {}'.format(outputcol), file=f)
        for ix,t in enumerate(tmp.index):
            print('{} \t \t {:.6f}'.format(t , tmp.loc[t]), file= f)
        print('\n', file= f)

with open(os.path.join(saveoutput,
            'info_about_counts_in_bg_data_whole_dataset.pickle'),'wb') as f2:
    pickle.dump((info_about_counts,Ntot,nullvalues),f2)
    
#%%
# only retain that part of the data that has education requirements
bgdatared = bgdata[~bgdata['MinEdu'].isnull()]

#%% 
'''
Add more info to the file
'''
# out of the ones with educational requirements, check how many also have 
# a low certainty SOC code, a salary and a skills vector
tmp = bgdatared[bgdatared['soc_score']>0.6]
#tmp = tmp[~tmp['MinSalary'].isnull()]
with open(os.path.join(saveoutput,'bgdata_initial_notes.txt'),'a') as f:
    B = len(tmp[(~tmp['converted_skills'].isnull()) & (~tmp['MinSalary'].isnull())])
    print('Number of ads with MinEdu, soc score > 0.6, skills and salary: {}',
          format(B), file = f)
    C = len(tmp[~tmp['converted_skills'].isnull()])
    print('Number of ads with MinEdu, soc score > 0.6 and skills: {}',
          format(C), file = f)
    D = len(tmp)
    print('Number of ads with MinEdu and soc score > 0.6: {}',
          format(D), file = f)
    
#%%
'''
# the following lines will show that MinEdu == 20 means SOC= 221 and 
# job title= orthodontist or specialty training
print(bgdatared[bgdatared['MinEdu']==20]['new_soc'])
print(bgdatared[bgdatared['MinEdu']==20]['title_processed'])

# now look to check whether orthodontists with soc = 221 could be assigned to 
# different educational requirements
orthodontist = bgdatared[bgdatared['title_processed']=='orthodontist']
orthodontist = orthodontist[(orthodontist['new_soc']==221)]

orthodontistfull = bgdata[bgdata['title_processed']=='orthodontist']
orthodontistfull = orthodontistfull[(orthodontistfull['new_soc']==221)]
'''

#%% 
'''
# save a reduced version of the dataset. It will save lots of time going forward
Only keep rows with at least one of the following columns being not null:
'MinEdu', 'MaxEdu', 'MinExp', 'MaxExp', 'MinSalary', 'MaxSalary'
'''
#condition1 = (~bgdata['MinEdu'].isnull()) | (~bgdata['MaxEdu'].isnull()) 
#condition2 = (~bgdata['MinExp'].isnull()) | (~bgdata['MaxExp'].isnull()) 
#| (~bgdata['MinSalary'].isnull()) | (~bgdata['MaxSalary'].isnull())
#print(condition1.sum(), condition2.sum())

#%%
#newcoi = ['Unnamed: 0', 'BGTJobId', 'JobDate', 'Employer',
#       'MinDegree', 'MaxDegree', 'MinEdu', 'MaxEdu',
#       'MinExp', 'MaxExp', 'MinSalary', 'MaxSalary',
#       'InternshipFlag', 'ApprenticeshipFlag',
#       'WorkFromHomeFlag', 'title_processed', 'converted_skills', 
#       'new_soc', 'soc_score', 'clusters', 'trans']

# save data with education requirements
#tmp = bgdata[condition1][newcoi]
#tmp.to_pickle(os.path.join(saveoutput,'bgdata_with_edu.zip'))

#%%
# save data with experience requirements
#tmp = bgdata[condition2][newcoi]
#tmp.to_pickle(os.path.join(saveoutput,'bgdata_with_exp.zip'))



    


