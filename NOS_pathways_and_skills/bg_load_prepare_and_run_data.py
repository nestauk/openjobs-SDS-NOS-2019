#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 09:19:05 2019

@author: stefgarasto

This script is for:
    1. Loading the data
    2. Extracting the good portion. That is, extracting the rows with
        - Min Edu
        - converted skills
        - good SOC code (BG SOC = new soc)
        - relevant SOC code (SOC code in any super-suite or in a specific one)
    3. Saving value counts to be plotted in bg_plot_counts_histograms.py when
        GOODDATA= True
    4. Reducing the categories for Min Edu
    5. Computing the London dummy variable
    6. Computing the average salary and filling in missing values with the mean
    7. Loading the pre-trained glove model

This is the first script that needs to be run
"""



#%%
#imports
from utils_bg import print_elapsed
import pandas as pd
import time
import pickle
import gensim
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
#from nltk.util import ngrams
 
from utils_bg import *

#%%
pd.options.mode.chained_assignment = None

#%% 
''' Load full dataset (this takes 23 minutes)
At some point might be good to compare with the load_all_bg_data function '''
print('Loading the dataset')
t0 = time.time()
for ix,year in enumerate(all_years):
    if ix == 0:
        bgdatared = pd.read_csv(filename.format(year))
    else:
        bgdatared = pd.concat((bgdatared, pd.read_csv(filename.format(year))))
        print(len(bgdatared))
print('Time in minutes: {:.4f}'.format((time.time()- t0)/60))


#%%
#Loading a pre-trained glove model into gensim
WHICH_GLOVE = 'glove.6B.100d' #'glove.6B.100d' #'glove.840B.300d', 
#glove.twitter.27B.100d

glove_dir = '/Users/stefgarasto/Local-Data/wordvecs/'

LOADGLOVE = True
if LOADGLOVE:
    print('Loading glove model')
    t0 = time.time()
    # load the glove model
    model = gensim.models.KeyedVectors.load_word2vec_format\
    (os.path.join(glove_dir, 'word2vec.{}.txt'.format(WHICH_GLOVE)))
    #model = api.load("glove-wiki-gigaword-100")  # load pre-trained word-vectors
    # from gensim-data
    #model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
    #word_vectors = model.wv
    print_elapsed(t0, 'loading the glove model')


#%% Load SOC codes of interest
#with open('/Users/stefgarasto/Google Drive/Documents/results/NOS/
#notes/soc_codes_list_for_bg.pickle','rb') as f:
#    total_socs4,total_socs3,soc4dist,soc3dist = pickle.load(f)

#%%
# need to define this here
SUPER = 'all'
SUPERS = ['all','engineering','management','financialservices','construction']
target_var = 'Exp3'

# convert SOCs from string to float
total_socs4_float = [float(t) for t in total_socs4]
soc4dist_float = {}
for key in soc4dist.keys():
    soc4dist_float[key] = [float(t) for t in soc4dist[key]]

'''
# first only keep those rows with a soc code that is relevant to at least one of
# the supersuites
'''
print('Starting process of data selection')
t0 = time.time()
if SUPER == 'all':
    bgdatared = bgdatared[bgdatared['SOC'].map(lambda x: x in total_socs4_float)]
    #astype(int).map(lambda x: str(x) in total_socs4)]
elif SUPER == 'none':
    bgdatared = bgdatared
else:
    bgdatared = bgdatared[bgdatared['SOC'].map(lambda x: x in soc4dist_float[SUPER])]
print_elapsed(t0, 'selecting the relevant SOCs')

#%%
''' Only keep rows with good 4 digits SOCs'''
print('Checking BG SOC = new soc')
t0 = time.time()
''' Check and keep the rows where BG soc = new soc'''
bgdatared['soc_flag']= bgsoc_is_newsoc(bgdatared)
# remove rows that don't agree
bgdatared = bgdatared[bgdatared['soc_flag']]
print_elapsed(t0, 'further reducing the dataset by good SOCs')

#%%
'''# only keep rows with skills'''
t0= time.time()
bgdatared = bgdatared[~bgdatared['converted_skills'].isnull()]
print_elapsed(t0, 'further reducing the dataset by good skills')

#%%
Nwhole_red = len(bgdatared)

#%% add the London flag
print('Computing the London dummy variable')
bgdatared['London'] = bgdatared['Region_Nation'].map(getlondonflag)

#%% add mean salary column
print('Computing the average salary')
bgdatared['MeanSalary_nan'] = (bgdatared['MinSalary'] + bgdatared['MaxSalary'])/2.0
bgdatared['MeanSalary'] = bgdatared['MeanSalary_nan'].fillna(
        bgdatared['MeanSalary_nan'].mean())

#%% compute skills and job title embeddings - at some point I will use every single
# one of these rows so I might as well
#%%
''' Compute the other feature vectors: first, job titles '''
# create file where to store words that are not in dictionary
GLOVE2FILE = False
if GLOVE2FILE:
    missing_jt_file = os.path.join(saveoutput,
                    'job_titles_not_in_dictionary_{}+{}.txt'.format(
                            WHICH_GLOVE,target_var))

    with open(missing_jt_file,'w') as f:
        print('Job titles not in the Twitter Glove dictionary', file= f)
        print('\n', file = f)
    
    recovered_jt_file = os.path.join(saveoutput,
            'job_titles_not_in_dictionary_{}_recovered_{}.txt'.format(WHICH_GLOVE,
                                          target_var))
    with open(recovered_jt_file,'w') as f:
        print('Job titles not in the Twitter Glove dictionary but recovered', file= f)
        print('\n', file = f)
    
    t0 = time.time()
    print('Computing the word embedding for job titles')
    bgdatared['title_embedding'] = bgdatared['title_processed'].map(
            lambda x: jt_to_vectors(x,model,missing_jt_file, recovered_jt_file))
else:
    t0 = time.time()
    print('Computing the word embedding for job titles')
    bgdatared['title_embedding'] = bgdatared['title_processed'].map(
            lambda x: jt_to_vectors_nofile(x,model))
    
print_elapsed(t0,'above')

#%%
''' Compute the other feature vectors: now, skills '''
#now for the skills
LOADSKILLS = True
SAVESKILLS = False
skills_ext = {'Exp':'Exp','Exp3':'Exp','Edu':'Edu','Eduv2':'Edu'}
if LOADSKILLS:
    print('Loading skills embedding')
    t0 =time.time()
    with open(os.path.join(savelocaloutput,'skills_embedding_{}_full.pickle'.format(
            WHICH_GLOVE)),'rb') as f:
        bgdatared['skills_embedding'] = pickle.load(f)
    print_elapsed(t0, 'above')
else:
    if GLOVE2FILE:
            # create file where to store words that are not in dictionary
        missing_jt_file = os.path.join(saveoutput,
                    'skills_not_in_dictionary_{}_full.txt'.format(WHICH_GLOVE))
        with open(missing_jt_file,'w') as f:
            print('Skills not in the Twitter Glove dictionary', file= f)
            print('\n', file = f)
        
        recovered_jt_file = os.path.join(saveoutput,
                    'skills_not_in_dictionary_{}_recovered_full.txt'.format(WHICH_GLOVE))
        with open(recovered_jt_file,'w') as f:
            print('Skills not in the Twitter Glove dictionary but recovered', file= f)
            print('\n', file = f)
        
        # get word embeddings
        t0 = time.time()
        print('Computing the word embedding for the skills')
        bgdatared['skills_embedding'] = bgdatared['converted_skills'].map(
                lambda x: skills_to_vectors(x,model,missing_jt_file, recovered_jt_file))

    else:
        # get word embeddings without saving anything to file
        t0 = time.time()
        print('Computing the word embedding for the skills')
        bgdatared['skills_embedding'] = bgdatared['converted_skills'].map(
                lambda x: skills_to_vectors_nofile(x,model))
    if SAVESKILLS:
        with open(os.path.join(savelocaloutput,'skills_embedding_{}_full.pickle'.format(
                WHICH_GLOVE)),'wb') as f:
            pickle.dump(bgdatared['skills_embedding'],f)
    print_elapsed(t0, 'above')
        
#%%
'''
# compute job title and skills embedding for unclassified jobs
t0 = time.time()
print('Computing the word embedding for job titles without requirements')
bgdatared['title_embedding'] = bgdatared['title_processed'
           ].map(lambda x: jt_to_vectors_nofile(x,model))
print_elapsed(t0, 'above')

#%% compute skills embedding
print('Computing the word embedding for the skills with requirements')
t0 = time.time()
bgdatared['skills_embedding'] = bgdatared['converted_skills'
           ].map(lambda x: skills_to_vectors_nofile(x,model))
print_elapsed(t0, 'above')
print_elapsed(t0, 'selecting the jobs to fill in')
'''

#%% ADD NEW TARGET VARIABLES 
print('Grouping the requirements into categories')
'''Split the educational/experience requirements into 4 (or 3) categories'''
t0 = time.time()
function_dict = {'Exp': group_exp, 'Exp3': group_exp3, 'Edu': group_edu, 
                 'Eduv2': group_eduv2}
target_col = {'Exp': 'MinExp', 'Exp3': 'MinExp', 'Edu': 'MinEdu', 
                 'Eduv2': 'MinEdu'}
#bgdatared['Exp'] = bgdatared['MinExp'].map(group_exp)
for ivar in ['Exp3','Eduv2']:
    bgdatared[ivar] = bgdatared[target_col[ivar]].map(
            function_dict[ivar])
print_elapsed(t0, 'above')

'''
# Up until here I gathered all the data that could potentially be useful - which
# is also all the data I want to estimate because it contains all those advert
with a relevant SOC code that was also estimated reliably with good probability
and with a skills vector. Below I split this according to the rows that have and
do not have the desired requirements.
'''
print('done')

#%% 
'''split into training and unknown dataset
# first extract the portion with the requirements'''
t0 = time.time()
if target_var in ['Edu','Eduv2']:
    condition1 = bgdatared['MinEdu'].isnull()
elif target_var in ['Exp','Exp3']:
    condition1 = bgdatared['MinExp'].isnull()
    
bgdatasmall= bgdatared[~condition1]
print_elapsed(t0, 'selecting data with requirements')

#%% 
'''now extract the portion of data without the requirements'''
FINALDATA = True
#if FINALDATA:   
#    t0 = time.time()
#    # keep the rows that I need to fill
#    #if target_var in ['Edu','Eduv2']:
#    #    bgdata_pred = bgdatared[bgdatared['MinEdu'].isnull()]
#    #elif target_var in ['Exp','Exp3']:
#    #    bgdata_pred = bgdatared[bgdatared['MinExp'].isnull()]
#    bgdata_pred = bgdatared[condition1]   
#    print_elapsed(t0, 'selecting rows without requirements')

    
#%% ADD NEW TARGET VARIABLES 

'''Split the educational/experience requirements into 4 (or 3) categories
print('Grouping the requirements into categories')
t0 = time.time()
function_dict = {'Exp': group_exp, 'Exp3': group_exp3, 'Edu': group_edu, 
                 'Eduv2': group_eduv2}
target_col = {'Exp': 'MinExp', 'Exp3': 'MinExp', 'Edu': 'MinEdu', 
                 'Eduv2': 'MinEdu'}
#bgdatared['Exp'] = bgdatared['MinExp'].map(group_exp)
bgdatasmall[target_var] = bgdatasmall[target_col[target_var]].map(
        function_dict[target_var])
print_elapsed(t0, 'above')
'''

#bgdatared = bgdatared[cols2keep]

#%%

'''
Get some counts info and save value counts for histogram
'''
GET_COUNTS = False
if GET_COUNTS:
    print('Get some counts of relevant variables')
    if target_var in ['Edu','Eduv2']:
        cols_to_count= ['MinEdu','SOC',target_var]
    elif target_var in ['Exp', 'Exp3']:
        cols_to_count= ['MinExp','SOC',target_var]
    else:
        raise ValueError
    Ns = {}
    Ns['reduced by all'] = len(bgdatasmall)
    t0 = time.time()
    # reduce to relevant SOC codes
    info_about_counts = {}
    for outputcol in cols_to_count:
        info_about_counts['Percentages for {}'.format(
                    outputcol)] = {}
        info_about_counts['Counts for {}'.format(
                    outputcol)] = {}
    # first for the "good" dataset as is, without removing some soc codes
    for outputcol in cols_to_count:
        #N = (~bgdatared['MinEdu'].isnull()).sum()
        tmp = bgdatasmall[outputcol].value_counts()
        N = tmp.sum()
        info_about_counts['Percentages for {}'.format(
                outputcol)]['reduced by all'] = tmp/N*100
        info_about_counts['Counts for {}'.format(
                outputcol)]['reduced by all'] = tmp
    
    # now get the joint counts of SOC with target variable
    # get frequency table
    joint_dist = pd.crosstab(bgdatasmall[target_var], 
                        bgdatasmall['SOCName'])    
    # normalise for each SOC code
    posterior = joint_dist/joint_dist.sum()
    
    # now get counts only for ROWS with relevant supersuites
    # get the value counts for all possibilities of reduced dataset
    for SUPER in SUPERS:
        if not SUPER == 'all':
            key = 'reduced by ' + SUPER
            bgdatatmp = bgdatasmall[bgdatasmall['SOC'].astype(int).map(
                    lambda x: x in soc4dist_float[SUPER])]
            Ns[key] = len(bgdatatmp)
            # get value counts
            for outputcol in cols_to_count:
                tmp = bgdatatmp[outputcol].value_counts()
                N = tmp.sum()
                info_about_counts['Percentages for {}'.format(
                        outputcol)][key] = tmp/N*100
                info_about_counts['Counts for {}'.format(
                        outputcol)][key] = tmp
    
    # save the results
    with open(os.path.join(saveoutput,
                'info_about_counts_in_bg_data_small_dataset_{}.pickle'.format(
                        target_var)),'wb') as f2:
        pickle.dump((info_about_counts,Ns,posterior),f2)
    
    bgdatatmp = None
    print_elapsed(t0, 'getting value counts')


#%% one-hot encoding of SOC code. Do it on the whole training dataset, since
# I don't really want to leave any SOC code out
'''do preprocessing to label SOC codes
t0 = time.time()
print('Computing One-hot encoding for the SOC codes')
#L = preprocessing.LabelEncoder().fit(X)
#enc = preprocessing.OneHotEncoder().fit(L.transform(X).reshape(-1,1))
#X = bgdatared['SOC'].values.reshape(-1,1)
enc = preprocessing.OneHotEncoder(categories = 'auto', handle_unknown ='ignore'
                                 ).fit(bgdatasmall['SOC'].values.reshape(-1,1))
#X=enc.transform(X)
print_elapsed(t0, 'above')
'''
#%
''' do preprocessing to label London variable'''
t0 = time.time()
print('Computing One-hot encoding for the London variable')
X_london = bgdatasmall['London'].values.reshape(-1,1)
enc_london = preprocessing.OneHotEncoder(categories = 'auto', 
                                         handle_unknown ='ignore'
                                         ).fit(X_london)
X_london = None
#X_london=enc_london.transform(X_london)
print_elapsed(t0, 'above')

#%%
''' 
Note: it takes approximately ? minutes overall
'''

#%% train the full hybrid model
TRAINHYBRID = False
if TRAINHYBRID:
    #%
    from bg_build_hybrid_model_fullcv import train_full_hybrid
    #%
    cv_models = train_full_hybrid(bgdatasmall, enc_london, 
                                  target_var = target_var)
    

#%%
CHECK_MODEL = False
if CHECK_MODEL:
    runfile(''.join(['/Users/stefgarasto/Google Drive/Documents/scripts/BG/',
                     'check_model_performance.py']), 
            wdir='/Users/stefgarasto/Google Drive/Documents/scripts/BG')


#%% 
'''this is to train the final model on all of the data
It also predicts the requirement on the missing rows and saves the results '''
TRAINFINAL = False
BOTH_TARGETS = True
SAVEPRED = False
if FINALDATA:
    #%
    if TRAINFINAL:
        try:
            from bg_build_hybrid_finalmodel_fullcv import train_full_hybrid_final
        except:
            from bg_build_hybrid_finalmodel_fullcv import train_full_hybrid_final
        #%
        final_models = train_full_hybrid_final(bgdatasmall, enc_london, 
                                  target_var = target_var)
    else:
        if BOTH_TARGETS:
            target_vars = ['Exp3','Eduv2']
        else:
            target_vars = [target_var]
        # model has been trained already - realod it
        match_th = 0.9
        for target_var0 in target_vars:
            with open(os.path.join(savelocaloutput,
                         'finalmodel_{:.0f}_results_hybrid_rfc_{}_{}.pickle'.format(
                         match_th*100,WHICH_GLOVE,target_var0)),'rb') as f:
                final_models = pickle.load(f)
        
            #% reload government based model
            if target_var0 in ['Edu','Eduv2']:
                with open(os.path.join(saveoutput,'government_based_model_Edu.pickle'), 'rb') as f:
                    official_soc_matches_final = pickle.load(f)
                    model_soc = final_models['model_soc']
                    # add the SOC that map to PhD level to the MAP model
                    for key in official_soc_matches_final:
                        if official_soc_matches_final[key] == 'Postgraduate':
                            model_soc[float(key)] = {'match': 'Postgraduate',
                                      'known accuracy' : 1}
            else:
                model_soc = final_models['model_soc']
            #'''# apply the final model to the rest of the data'''
            ## first, load the out of data SOCs (unique to the NOS dataset)
            #from bg_solve_specific_soc_codes import matches_oobsoc_to_soc
            ## turn them into float numbers
            #matches_oobsoc_to_soc2 = {}
            #for key in matches_oobsoc_to_soc.keys():
            #    new_key = float(matches_oobsoc_to_soc[key])
            #    matches_oobsoc_to_soc2[float(key)] = {'match': new_key,
            #                           'known accuracy': 1.0}
            
            #% apply model
            t0 = time.time()
            bgdatared['Target_pred_'+target_var0] = 'empty'
            bgdatared['Target_prob_'+target_var0] = 0
            # assign each known soc code to its match
            for isoc in model_soc:
                locations = bgdatared['SOC']==isoc
                # assign the predicted category
                bgdatared['Target_pred_'+target_var0][locations] = model_soc[isoc]['match']
                # assign the probability
                bgdatared['Target_prob_'+target_var0][locations] = model_soc[isoc]['known accuracy']
            print_elapsed(t0, 'classified socs via MAP')
            # assign each known job title to its match
            for isoc in final_models['model_title']:
                locations = bgdatared['title_processed']==isoc
                # assign the predicted category
                bgdatared['Target_pred_'+target_var0][locations] = final_models[
                        'model_title'][isoc]['match']
                # assign the probability
                bgdatared['Target_prob_'+target_var0][locations] = final_models[
                        'model_title'][isoc]['known accuracy']
            print_elapsed(t0, 'Classified titles via MAP')
            
            jobs_unclassified = bgdatared['Target_pred_'+target_var0]== 'empty'
            # extract features from the unclassified data
            x = get_all_features(bgdatared[jobs_unclassified], 
                                      final_models['SOC_enc'], enc_london)
            print_elapsed(t0, 'computing features')
            bgdatared['Target_pred_'+target_var0][jobs_unclassified] = final_models['results'][
                                'ROS+RFC']['classifier'].predict(x)
            bgdatared['Target_prob_'+target_var0][jobs_unclassified] = final_models['results'][
                                'ROS+RFC']['classifier'].predict_proba(x).max(axis = 1)
            print_elapsed(t0, 'classified jobs via RFC')
        
            # define columns to save
            cols_to_save = ['Unnamed: 0', 'BGTJobId', 'Target_pred_'+target_var0,
                            'Target_prob_'+target_var0]
                            #['Unnamed: 0', 'BGTJobId', 'JobDate', 'SOC', 'SOCName', 
                            #'PayFrequency', 'SalaryType',
                            #'clusters', 'title_processed', 'London',
                            #'MeanSalary_nan', 'MeanSalary', 'Target_pred', 'Target_prob',
                            #'title_embedding', 'skills_embedding']
            
            # save new dataset with predictions
            if SAVEPRED:
                bgdatared[cols_to_save].reset_index().to_pickle(os.path.join(savelocaloutput,
                       'prediction_missing_data_for_{}.pickle'.format(target_var0)))
            #bgdata_pred[cols_to_keep].reset_index.to_pickle(os.path.join(savelocaloutput,
            #           'prediction_missing_data_for_{}.pickle'.format(target_var)))
        
    
