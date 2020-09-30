#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:56:00 2019

@author: stefgarasto

This script is meant to train the final model, 
that is a random forest classifier trained on the whole
available data
 
"""



#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
#from scipy import interp
from sklearn import ensemble
from sklearn.model_selection import StratifiedKFold#, StratifiedShuffleSplit
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
#from imblearn.pipeline import make_pipeline
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import f1_score
from sklearn.dummy import DummyClassifier
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
#from copy import deepcopy
#from imblearn.ensemble import BalancedRandomForestClassifier
import time
from sklearn.metrics import precision_recall_fscore_support

#%% Benchmark classifiers
from utils_bg import saveoutput, socnames_dict, print_elapsed, get_all_features
from utils_bg import savelocaloutput
#%%
def train_full_hybrid_final(bgdatasmall, enc_london, classifiers = None,
                     SAVERES = True, SAVEFIG = False, match_th = .9, 
                     extra_args = {'SUPER': 'all'},
                    WHICH_GLOVE = 'glove.6B.100d', target_var = 'Edu'):
    
    #%%
    print('Setting up the classifier')
    # set up the classifiers
    if not classifiers:
        classifiers = [['RFC', ensemble.RandomForestClassifier(n_estimators=100, 
                                                         n_jobs = -1)]]
    savelocaloutput = '/Users/stefgarasto/Local-Data/Results/SDS-BG/'

    #%%
    # set up all the possible samplers
    samplers = [
        ['ROS', RandomOverSampler()]
    ]
    
    #%%
    classes_dict = {'Edu': ['Pregraduate','Graduate','Postgraduate'],
                    'Eduv2': ['Pregraduate','Graduate','Postgraduate'],
                    'Exp': ['Entry-level','Junior','Senior','Expert'],
                    'Exp3': ['Entry-level','Mid-level','Senior-level']}
    classes = classes_dict[target_var]
    
    #%%
    # build the whole model on three different cross validatation splits
    # however, use a partition-type cross validation
    
    
    '''
    how it's going to go
    - classify easy SOCs. OK
    - classify easy titles. OK
    - remove classified socs and titles from training data. OK
    - rebuild SOC one-hot encoder. OK
    - oversample training data.
    - fit classifier on training data. 
    - save final classifier. 
    
    '''
    
    #%%
    print('Starting the model training phase')
    t0 = time.time()
    ''' 
    Compute the posterior of Edu given SOC codes.
    '''
    # get frequency table
    joint_dist = pd.crosstab(bgdatasmall[target_var], 
                        bgdatasmall['SOC'])
    
    # normalise for each SOC code
    posterior = joint_dist/joint_dist.sum()
    
    # normalise by counts of MinEdu
    likelihood = (joint_dist.T)/(joint_dist.T.sum())
    likelihood = (likelihood.T)    
    
    # now get the prior
    prior = bgdatasmall[target_var].value_counts()
    prior = (prior/prior.sum()).values
    
    #%
    ''' Collect and print to file all the SOC that are matched to a category more 
    than 90% of the time. Store the results in model_soc4_to_edu_cat1
    # remove matched socs that are only counted once
    '''
    matched_socs = posterior.columns[posterior.max()>match_th]
    model_soc4_to_edu_cat1 = {}
    for isoc in matched_socs:
        matched_to = posterior.index[posterior[isoc]>match_th].values[0]
        model_soc4_to_edu_cat1[isoc] = {'match': matched_to, 
                        'known accuracy': posterior.loc[matched_to][isoc]}
    ja_tot = 0
    if SAVERES:
        file2use= ''.join(['finalmodel_occupations_matched_to_{}_category1_'.format(
                                            target_var.lower()),
                           '{:.0f}_20190729.txt'.format(100*match_th)])
        with open(os.path.join(saveoutput,file2use), 'w') as f:
            for isoc in matched_socs:
                matched_to = posterior.index[posterior[isoc]>match_th].values[0]
                print('\n',file=f)
                print(
                ''.join(['\'{}\' ({}) was matched to \'{}\' {:.2f}% of '.format(
                        socnames_dict[int(isoc)], 
                        int(isoc),
                        matched_to,
                        100*posterior.loc[matched_to][isoc]),
                    'the time (out of {} occurrences)'.format(
                        joint_dist[isoc].sum())]), file= f)
                ja_tot += joint_dist[isoc].sum()
            print('\n', file =f)
            print('Number of occupations matched: {}'.format(len(matched_socs)),
                  file=f)
            print('\n', file = f)
            print(''.join(['Total job adverts covered by this model: ',
                           '{} out of {} ({:.2f}%)'.format(
                    ja_tot, joint_dist.sum().sum(),
                    100*ja_tot/joint_dist.sum().sum())]),
                  file=f)    

    #%
    '''
    for each SOC that has not been matched get the 10 most common job titles and
    classify them as above. Store the results in model_titles_to_edu_cat2
    '''
    soc_by_jt = pd.crosstab(bgdatasmall['title_processed'], bgdatasmall['SOC'])
    jt_nb_by_socs = (soc_by_jt>0).sum()
    groups = bgdatasmall.groupby('title_processed')
    non_matched_socs = list(set(jt_nb_by_socs.index) - set(matched_socs))
    model_titles_to_edu_cat2 = {}
    counter =0
    ja_tot = 0
    for isoc in non_matched_socs:
        # get the 10 most common job titles
        common_jt = soc_by_jt.T.loc[isoc].sort_values(ascending = False)[:10]
        for ijt in common_jt.index:
            # get the relevant rows for this job title
            group = groups.get_group(ijt)
            # distribution by educational category
            edu_cat_dist = group[target_var].value_counts()
            # normalise to get probabilities
            edu_cat_dist = edu_cat_dist/edu_cat_dist.sum()
            if edu_cat_dist.max()>match_th:
                # also, check for any clashes with the model of education by soc
                soc_dist = group['SOC'].value_counts()
                soc_dist = soc_dist/soc_dist.sum()
                matched_to = edu_cat_dist[edu_cat_dist>match_th].index.values[0]
                soc_intersect= list(set(soc_dist.index).intersection(
                        set(matched_socs)))
                if len(soc_intersect)>0:
                    # if this tile sometimes belong to a soc that has already been
                    # assigned, check that the assignments are consistent
                    for isoc2 in soc_intersect:
                        #print(matched_to, model_soc4_to_edu_cat1[isoc2])
                        if matched_to != model_soc4_to_edu_cat1[isoc2]['match']:
                            # there is one instance of this problem and it's a case
                            # where the association by job title seems better
                            # also, it concerns only one job in the training data
                            # so I'll override the association by soc code in this 
                            # instance
                            print('There\'s a clash')
                            counter+=1
                model_titles_to_edu_cat2[ijt] = {'match': matched_to,
                                'known accuracy': edu_cat_dist.loc[matched_to]}

    if SAVERES:
        file2use = ''.join(['finalmodel_titles_matched_to_{}_category2_'.format(
                            target_var.lower()),
                            '{:.0f}_20190729.txt'.format(100*match_th)])
        with open(os.path.join(saveoutput,file2use), 'w') as f:
            for isoc in non_matched_socs:
                # get the 10 most common job titles
                common_jt = soc_by_jt.T.loc[isoc].sort_values(ascending = False)[:10]
                for ijt in model_titles_to_edu_cat2.keys():
                    # get the relevant rows for this job title
                    group = groups.get_group(ijt)
                    print('\n',file=f)
                    print(
                    ''.join(['\'{}\' was matched to \'{}\' {:.2f}% '.format(
                                ijt, 
                                model_titles_to_edu_cat2[ijt]['match'],
                                100*model_titles_to_edu_cat2[ijt]['known accuracy']),
                             'of the time (out of {} occurrences)'.format(
                                len(group))]), file= f)
                    ja_tot += len(group)
                    
            print('\n', file =f)
            print('Number of occupations matched: {}'.format(
                    len(model_titles_to_edu_cat2)),
                  file=f)
            print('\n', file = f)
            print('Total job adverts covered by this model: {} out of {} ({:.2f}%)'.format(
                    ja_tot, joint_dist.sum().sum(),100*ja_tot/joint_dist.sum().sum()),
                  file=f)
        
    
    print_elapsed(t0, 'building the MAP model for SOCs and titles')
    
    #%
    '''
    Now we move to the part of the model that is trained using Random Forest 
    Classifiers.
    First, extract the part of the dataset that has not been classified yet
    '''
    # training data
    jobs_classified_train = bgdatasmall['SOC'].isin(list(model_soc4_to_edu_cat1.keys(
            ))) | bgdatasmall['title_processed'].isin(
                                list(model_titles_to_edu_cat2.keys()))
    bgleft_train = bgdatasmall[~jobs_classified_train]
    
    
    #%
    '''
    extract all features from the data that is left (training and testing).
    First, rebuild the encoder for the SOC codes, since some of them are not in 
    use anymore. The encoder needs to be fit on the training data. Note that now
    there might be some SOC codes in the test data that don't appear on the
    training data
    '''
    print('Computing One-hot encoding for the remaining SOC codes')
    t0 =time.time()
    enc_left = preprocessing.OneHotEncoder(categories = 'auto', handle_unknown='ignore'
                                     ).fit(bgleft_train['SOC'].values.reshape(-1,1))
    print_elapsed(t0, 'above')
    
    #%
    print('Getting the features for the second part of the hybrid model')
    t0 = time.time()
    x_train = get_all_features(bgleft_train, enc_left, enc_london)
    #xunit = StandardScaler().fit_transform(x)
    print_elapsed(t0,'task above')
    
    # add scaling? 
    #% get the classification targets
    y_train = bgleft_train[target_var].values
    
    #%%
    results = {}
    for name, sampler in samplers[:1]:
        for cname, classifier in classifiers:
            fullname = name + '+' + cname
            t0 = time.time()
            clf = clone(classifier)
            #ensemble.RandomForestClassifier(n_estimators=100, n_jobs = -1)
            X_res, y_res = sampler.fit_resample(x_train,y_train)
            print(clf)
            t00 = time.time()
            clf.fit(X_res, y_res)
            train_time = time.time() - t00
            print("train time: %0.3fs" % train_time)
            clf_descr = str(clf).split('(')[0]

            # add everything to the dictionary
            results[fullname] = {'name':clf_descr, 'classifier': clf}
            # not saving anymore bc of size: 'classifier': clf, 
            
            print_elapsed(t0, 'training RFC using sampler {}'.format(name))
            

    #%% append the result to a list (one element for each partition)
    final_models ={'model_soc': model_soc4_to_edu_cat1, 
                      'model_title': model_titles_to_edu_cat2, 
                      'classes': classes, 'results': results,
                      'jobs_classified_MAP': jobs_classified_train, 
                      'SOC_enc': enc_left, 'extra_args': extra_args}
        # not saving these anymore: 'train_indices': train, 'test_indices': test
        # 'results_dummy': results_dummy, 'y_pred_MAP' : y_pred_MAP, 

    #%%
    if SAVERES:
        try:
            with open(os.path.join(savelocaloutput,
                     'finalmodel_{:.0f}_results_hybrid_rfc_{}_{}.pickle'.format(
                     match_th*100,WHICH_GLOVE,target_var)),'wb') as f:
                pickle.dump(final_models,f)
        except:
            1
        
    return final_models


if __name__ == "__main__":
    cv_models = train_full_hybrid_final(bgdatasmall, enc_london, target_var = target_var)