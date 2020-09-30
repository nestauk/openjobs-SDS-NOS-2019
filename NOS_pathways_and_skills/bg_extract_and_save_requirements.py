#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:27:17 2019

@author: stefgarasto
"""

#%%
'''
This script require "bg_load_prepare_and_run" to have been run 
with FINALDATA=True

Next steps:
1. reload predictions and already filled in data
2. Only keep those rows for which the requirements estimates are reliable
enough (i.e. more than 50% certainty)
3. For each SOC, SC and SOC + SC extract all the job adverts with same tags
4. For each tag, save:
    a. collection counter for qual 
    b. collection counter for exp
    c. full array of mean salaries / collection counter for mean salaries?
    d. collection counters for skills
    e. collection counters for job titles
    f. London variable
    g. distribution of word embeddings? Average of word embeddings?

5. 
'''

import time
import os
from utils_bg import *
from utils_nlp import *
from utils_skills_clusters import load_and_process_clusters
import numpy as np
from collections import Counter

#%
# Reload estimated missing data
t0 = time.time()
bgdata_pred_exp = pd.read_pickle(os.path.join(savelocaloutput,
               'prediction_missing_data_for_{}.pickle'.format('Exp3')))
bgdata_pred_edu = pd.read_pickle(os.path.join(savelocaloutput,
               'prediction_missing_data_for_{}.pickle'.format('Eduv2')))
print_elapsed(t0, 'loading estimates for missing data')

#%% get the boolean indices
t0 = time.time()
condition_exp = bgdatared['MinExp'].isnull()
condition_edu = bgdatared['MinEdu'].isnull()
print_elapsed(t0, 'getting boolean indicators')

#% add the estimates
bgdatared['PredExp'] = 'unknown'
bgdatared['PredEdu'] = 'unknown'

#%%
# Exp: from the model, predictions and probabilities
#JOINPRED = False
#if JOINPRED:
#    bgdatared2 = bgdatared.join((bgdata_pred_exp.set_index('index'))[['Target_pred_Exp3',
#               'Target_prob_Exp3']])
#else:
#    bgdatared['Target_pred_Exp3'] = bgdata_pred_exp['Target_pred_Exp3']
#    bgdatared['Target_prob_Exp3'] = bgdata_pred_exp['Target_prob_Exp3']
    
#%% replace the ones that we can get from the data
bgdatared['Target_pred_Exp3'][~condition_exp] = bgdatared['Exp3'][~condition_exp]
bgdatared['Target_prob_Exp3'][~condition_exp] = 1

#%% Edu: from the model, predictions and probabilities
#t0 = time.time()
#if JOINPRED:
#    bgdatared = bgdatared.join((bgdata_pred_edu.set_index('index'))[['Target_pred_Eduv2',
#               'Target_prob_Eduv2']])
#else:
#    bgdatared['Target_pred_Edu2'] = bgdata_pred_exp['Target_pred_Edu2']
#    bgdatared['Target_prob_Edu2'] = bgdata_pred_exp['Target_prob_Edu2']
#print_elapsed(t0, 'adding education')
    
#%% replace the ones that we can get from the data
bgdatared['Target_pred_Eduv2'][~condition_edu] = bgdatared['Eduv2'][~condition_edu]
bgdatared['Target_prob_Eduv2'][~condition_edu] = 1

#% only keep the reliable estimates (including the existing ones that are well 
# understood from the model?)
t0 = time.time()
bgdata_final = bgdatared[(bgdatared['Target_prob_Eduv2']>.5) & 
                         (bgdatared['Target_prob_Exp3']>.5)]
print_elapsed(t0, 'keeping rows we are certain of')

#% finally, drop irrelevant columns and rename other (?)
t0 = time.time()
cols2keep = ['Unnamed: 0', 'BGTJobId', 'JobDate', 'SOC', 'SOCName', 
             'PayFrequency', 'SalaryType',
             'title_processed', 'converted_skills', 'clusters', 'London', 
             'MeanSalary_nan', 'MeanSalary', 'title_embedding', 'skills_embedding',
             'Exp3', 'Eduv2', 'Target_pred_Exp3', 'Target_prob_Exp3',
             'Target_pred_Eduv2', 'Target_prob_Eduv2']
bgdata_final = bgdata_final[cols2keep]
bgdata_final = bgdata_final.rename(columns = {'Target_pred_Eduv2': 'myEdu',
                                              'Target_pred_Exp3': 'myExp'})
print_elapsed(t0,'reducing the final dataset')

#%%
# create skill clusters
clus_names, comparison_vecs, skill_cluster_vecs = load_and_process_clusters(model)


## In[199]:    
# match average skills embedding to closest cluster
t0 = time.time()
#ja_edu_v_clus = {}
bgdata_final['best_cluster'] = bgdata_final['skills_embedding'].map(
        lambda x: highest_similarity(x, comparison_vecs, clus_names))

print_elapsed(t0,'assigning skills clusters')

#%%
def eval_clusters(x):
    if isinstance(x, str):
        skills = eval(x)
    else:
        skills = eval(x.values[0])
    return skills
#%%
def match_all_skills(x,model, comparison_vecs, clus_names):
    full_skills= skills_to_vectors_full_nofile(x, model)
    tmp_clus = []
    for iskill in range(full_skills.shape[0]):
        tmp_clus.append(highest_similarity(full_skills[iskill], comparison_vecs, clus_names)) 
    return tmp_clus
#%%
def get_top_cluster(x):
    xtmp = x[0] # assuming the first skill is more important
    x = Counter(x).most_common()
    if x[0][1]>1:
        return x[0][0]
    else:
        return xtmp #highest_similarity(test_skills, comparison_vecs, clus_names)

#%%
bgdata_final['clusters2'] = bgdata_final['clusters'].map(eval_clusters)

#%%
locations = bgdata_final['clusters2'].map(lambda x: len(x)==0)
bgdata_final['clusters2'][locations]= bgdata_final['converted_skills'][locations
            ].map(lambda x: match_all_skills(x,model, comparison_vecs,clus_names)
            )#.map(get_top_cluster)

#%%
bgdata_final['best_cluster2'] = bgdata_final['clusters2'].map(get_top_cluster)

##%%
#bgdata_final['best_cluster2'][bgdata_final['best_cluster2']=='empty'
#             ] = bgdata_final['best_cluster'][bgdata_final['best_cluster2']=='empty']

#%%
cols2match = ['myExp','myEdu','MeanSalary_nan','best_cluster2']

# In[198]:
'''
assign requirements to each skill cluster

For each tag, save:
    a. collection counter for qual 
    b. collection counter for exp
    c. collection counter for SOCs
    d. full array of mean salaries / collection counter for mean salaries?
    e. collection counters for skills
    f. collection counters for job titles
    g. collection counter for London variable
    h. distribution of word embeddings? Average of word embeddings?

    
'''
t0 = time.time()
JA2CLUS = True
if JA2CLUS:
    ix = 0  
    cols_v_clus = {}
    bg_groups = bgdata_final.groupby(by = 'best_cluster2')
    for name,group in bg_groups:
        cols_v_clus[name] = {}
        if not len(group):
            print(name)
            continue
        for col in ['myExp','myEdu','SOC']:
            cols_v_clus[name][col] = Counter(group[col])#.value_counts()
            #try:
            cols_v_clus[name][col + '-peak'] = cols_v_clus[name][col].most_common(
                            )[0][0]  #cols_v_clus[name][col].idxmax()
            #except:
            #    cols_v_clus[name][col + '-peak'] = 'unknown'
        # now the mean salary
        tmp = group['MeanSalary_nan'].values
        cols_v_clus[name]['Salary'] = tmp[~np.isnan(tmp)]
        cols_v_clus[name]['Salary-peak'] = np.nanmean(group['MeanSalary_nan'])
        # job titles
        cols_v_clus[name]['title_processed'] = Counter(group['title_processed'])
        # skills
        # first concatenate and trasnform into a list
        tmp = eval(''.join(list(group['converted_skills'].values)).replace('][',','))
        cols_v_clus[name]['converted_skills'] = Counter(tmp)
        # london variable
        cols_v_clus[name]['London'] = Counter(group['London'])
        # average skills embedding
        cols_v_clus[name]['average_skills_embedding'] = group['skills_embedding'].mean()
        ix+=1
        if ix%30 == 29:
            print('Got to skill cluster number {}'.format(ix))
print_elapsed(t0, 'matching clusters to requirements')
        
## In[]:

'''
assign requirements to each occupation
'''
ix = 0

t0 = time.time()
bg_occ_groups = bgdata_final.groupby(by = 'SOC')
cols_v_occ = {}

for name,group in bg_occ_groups:
    cols_v_occ[name] = {}
    if not len(group):
        print(name)
        continue
    for col in ['myExp','myEdu','best_cluster2']:
        cols_v_occ[name][col] = Counter(group[col])#.value_counts()
        #try:
        cols_v_occ[name][col + '-peak'] = cols_v_occ[name][col].most_common(
                            )[0][0] #cols_v_occ[name][col].idxmax()
        #except:
        #    cols_v_occ[name][col + '-peak'] = 'unknown'
    # now the mean salary
    tmp = group['MeanSalary_nan'].values
    cols_v_occ[name]['Salary'] = tmp[~np.isnan(tmp)]
    cols_v_occ[name]['Salary-peak'] = np.nanmean(group['MeanSalary_nan'])
    # job titles
    cols_v_occ[name]['title_processed'] = Counter(group['title_processed'])
    # skills
    # first concatenate and trasnform into a list
    tmp = eval(''.join(list(group['converted_skills'].values)).replace('][',','))
    cols_v_occ[name]['converted_skills'] = Counter(tmp)
    # london variable
    cols_v_occ[name]['London'] = Counter(group['London'])
    # average skills embedding
    cols_v_occ[name]['average_skills_embedding'] = group['skills_embedding'].mean()
    ix+=1
    if ix%30 == 29:
        print('Got to SOC number {}'.format(ix))
        
print_elapsed(t0, 'matching SOC to requirements')

## In[]:

'''
assign requirements to each occupation+cluster combo
'''
ix = 0
#%
t0 = time.time()
bg_groups = bgdata_final.groupby(by = ['SOC', 'best_cluster2'])
cols_v_occ_and_clus = {}
ix=0
for name0,group in bg_groups:
    # create join name
    name = '+'.join([str(t) for t in name0])
    #if name[0] not in cols_v_occ_and_clus:
    #    cols_v_occ_and_clus[name[0]] = {}
    #cols_v_occ_and_clus[name[0]][name[1]] = {}
    cols_v_occ_and_clus[name]= {}
    if not len(group):
        print(name)
        continue
    for col in ['myExp','myEdu']:
        #cols_v_occ_and_clus[name[0]][name[1]][col] = Counter(group[col])#.value_counts()
        cols_v_occ_and_clus[name][col] = Counter(group[col])
        #try:
        cols_v_occ_and_clus[name][col + '-peak'] = cols_v_occ_and_clus[name][
                col].most_common()[0][0] #cols_v_occ_and_clus[name][col].idxmax()
        #except:
        #    cols_v_occ_and_clus[name][col + '-peak'] = 'unknown'
    # now the mean salary
    tmp = group['MeanSalary_nan'].values
    cols_v_occ_and_clus[name]['Salary'] = tmp[~np.isnan(tmp)]
    cols_v_occ_and_clus[name]['Salary-peak'] = np.nanmean(
            group['MeanSalary_nan'])
    # job titles
    cols_v_occ_and_clus[name]['title_processed'] = Counter(
            group['title_processed'])
    # skills
    # first concatenate and trasnform into a list
    tmp = eval(''.join(list(group['converted_skills'].values)).replace('][',','))
    cols_v_occ_and_clus[name]['converted_skills'] = Counter(tmp)
    # london variable
    cols_v_occ_and_clus[name]['London'] = Counter(group['London'])
    # average skills embedding
    cols_v_occ_and_clus[name]['average_skills_embedding'] = group[
                                                'skills_embedding'].mean()
    ix+=1
    if ix%300 == 299:
        print('Got to SOC+clus number {}'.format(ix))
        
print_elapsed(t0, 'matching SOC+clusters to requirements')

#%% save all of these dictionaries:
# cluster
with open(os.path.join(saveoutput,'cols_v_clus3.pickle'), 'wb') as f:
    pickle.dump(cols_v_clus,f)

#%% SOC
with open(os.path.join(saveoutput,'cols_v_occ3.pickle'), 'wb') as f:
    pickle.dump(cols_v_occ,f)
    
#%% SOC+cluster
with open(os.path.join(saveoutput,'cols_v_occ_and_clus3.pickle'), 'wb') as f:
    pickle.dump(cols_v_occ_and_clus,f)


