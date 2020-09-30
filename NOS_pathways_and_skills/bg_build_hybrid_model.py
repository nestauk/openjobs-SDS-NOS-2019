#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 18:02:03 2019

@author: stefgarasto

This scripts builds the "hybrid model" whereby some educational requirements
are estimated by MAP on the SOC codes (most common classification per SOC code,
when the MAP is > a certain threshold); some by MAP on common job titles; some
by random forest classification. Only the latter is done via cross validation

It assumes that bg_load_and_prepare_data has been run

NOTE: retrain SOC encoder by cv split - Other adjustments, 
not sure they're needed though (see plan)
"""
#TODO: see note above

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
from copy import deepcopy
#from imblearn.ensemble import BalancedRandomForestClassifier
import time

#%% Benchmark classifiers
from utils_bg import saveoutput, socnames_dict, print_elapsed, get_all_features
from utils_bg import benchmark, benchmark_scores



'''
#%% first reduce the dataset further to keep only SOC codes relevant to the 
# super-suites
t0 = time.time()
SUPER = 'all'
if SUPER == 'all':
    bgdatasmall = bgdatared[bgdatared['SOC'].astype(int).map(
                lambda x: str(x) in total_socs4)]
elif SUPER == 'none':
    bgdatasmall = bgdatared
else:
    bgdatasmall = bgdatared[bgdatared['SOC'].astype(int).map(
                lambda x: str(x) in soc4dist[SUPER])]
print_elapsed(t0, 'selecting SOCs relevant to super-suites')
'''
#%%
def train_full_hybrid(bgdatasmall, enc_london, 
                     SAVERES = False, SAVEFIG = False, match_th = .9):

    #%%
    t0 = time.time()
    ''' 
    Compute the posterior of Edu given SOC codes.
    '''
    # get frequency table
    joint_dist = pd.crosstab(bgdatasmall['Edu'], 
                        bgdatasmall['SOC'])
    # the baseline accuracy of any model is the proportion of the biggest class
    baseline_accuracy = np.around((joint_dist.sum())/(joint_dist.sum().sum()).max())
    
    # normalise for each SOC code
    posterior = joint_dist/joint_dist.sum()
    
    # normalise by counts of MinEdu
    likelihood = (joint_dist.T)/(joint_dist.T.sum())
    likelihood = (likelihood.T)
    # this should correspond to the MultinomionalNB with alpha=0, fit_prior = False
    #  so that likelihood == np.exp(clf_basic.feature_log_prob_)
    
    
    # now get the prior
    prior = bgdatasmall['Edu'].value_counts()
    prior = (prior/prior.sum()).values
    
    #%%
    ''' Collect and print to file all the SOC that are matched to a category more 
    than 90% of the time
    '''
    matched_socs = posterior.columns[posterior.max()>match_th]
    model_soc4_to_edu_cat1 = {}
    ja_tot = 0
    if SAVERES:
        file2use = 'occupations_matched_to_edu_category1_{:.0f}_20190729.txt'.format(
                    100*match_th)
    else:
        file2use = 'tmp.txt'
        
    with open(os.path.join(saveoutput,file2use), 'w') as f:
        for isoc in matched_socs:
            matched_to = posterior.index[posterior[isoc]>match_th].values[0]
            model_soc4_to_edu_cat1[isoc] = {'match': matched_to, 
                                  'known accuracy': posterior.loc[matched_to][isoc]}
            print('\n',file=f)
            print(
            ''.join(['\'{}\' ({}) was matched to \'{}\' {:.2f}% of the time (out ',
                     'of {} occurrences)'.format(
                socnames_dict[int(isoc)], 
                int(isoc),
                matched_to,
                100*posterior.loc[matched_to][isoc],
                joint_dist[isoc].sum())]), file= f)
            ja_tot += joint_dist[isoc].sum()
        print('\n', file =f)
        print('Number of occupations matched: {}'.format(len(matched_socs)),
              file=f)
        print('\n', file = f)
        print('Total job adverts covered by this model: {} out of {} ({:.2f}%)'.format(
                ja_tot, joint_dist.sum().sum(),100*ja_tot/joint_dist.sum().sum()),
              file=f)
    
    if SAVERES:
        with open(os.path.join(saveoutput,'model_soc4_to_edu_cat1_{:.0f}.pickle'.format(
            match_th*100)),'wb') as f:
            pickle.dump(model_soc4_to_edu_cat1,f)
        
    '''
    The procedure above assigns 29 SOCs that correspond to 47% of the job adverts
    '''
    
    #%% 
    '''
    for each SOC that has not been matched get the 10 most common job titles and
    classify them as above
    '''
    soc_by_jt = pd.crosstab(bgdatasmall['title_processed'], bgdatasmall['SOC'])
    jt_nb_by_socs = (soc_by_jt>0).sum()
    groups = bgdatasmall.groupby('title_processed')
    non_matched_socs = list(set(jt_nb_by_socs.index) - set(matched_socs))
    model_titles_to_edu_cat2 = {}
    counter =0
    ja_tot = 0
    if SAVERES:
        file2use = 'titles_matched_to_edu_category2_{:.0f}_20190729.txt'.format(
                    100*match_th)
    else:
        file2use = 'tmp.txt'
        
    with open(os.path.join(saveoutput,file2use), 'w') as f:
        for isoc in non_matched_socs:
            # get the 10 most common job titles
            common_jt = soc_by_jt.T.loc[isoc].sort_values(ascending = False)[:10]
            for ijt in common_jt.index:
                # get the relevant rows for this job title
                group = groups.get_group(ijt)
                # distribution by educational category
                edu_cat_dist = group['Edu'].value_counts()
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
                    print('\n',file=f)
                    print(
                    ''.join(['\'{}\' was matched to \'{}\' {:.2f}% of the time (out ',
                             'of {} occurrences)'.format(
                        ijt, 
                        matched_to,
                        100*edu_cat_dist.loc[matched_to],
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
    
    if SAVERES:
        with open(os.path.join(saveoutput,
                'model_titles_to_edu_cat2_{:.0f}.pickle'.format(
                match_th*100)),'wb') as f:
            pickle.dump(model_titles_to_edu_cat2,f)
        
    '''
    The procedure above assigns 309 titles that correspond to 9% of the job adverts
    '''
    
    print_elapsed(t0, 'building the MAP model for SOCs and titles')
    
    #%%
    '''
    Now we move to the part of the model that is trained using Random Forest 
    Classifiers.
    First, extract the part of the dataset that has not been classified yet
    '''
    
    jobs_classified = bgdatasmall['SOC'].isin(list(model_soc4_to_edu_cat1.keys())
                        ) | bgdatasmall['title_processed'].isin(
                                list(model_titles_to_edu_cat2.keys()))
    bgdataleft = bgdatasmall[~jobs_classified]
    
    #%%
    '''
    extract all features from bgdatalef.
    First, rebuild the encoder for the SOC codes, since some of them are not in 
    use anymore
    '''
    print('Computing One-hot encoding for the remaining SOC codes')
    t0 =time.time()
    enc_left = preprocessing.OneHotEncoder(categories = 'auto', handle_unknown='ignore'
                                     ).fit(bgdataleft['SOC'].values.reshape(-1,1))
    print_elapsed(t0, 'above')
    
    #%%
    print('Getting the features for the second part of the hybrid model')
    t0 = time.time()
    x = get_all_features(bgdataleft,enc_left,enc_london)
    #xunit = StandardScaler().fit_transform(x)
    print_elapsed(t0,'task above')
    
    #%% get the classification targets
    y = bgdataleft['Edu'].values
    
    #%%
    LW = 2
    RANDOM_STATE = 42
    
    class DummySampler:
    
        def sample(self, X, y):
            return X, y
    
        def fit(self, X, y):
            return self
    
        def fit_resample(self, X, y):
            return self.sample(X, y)
    
    #%%
    # set up the classifier
    classifiers = [['RFC', ensemble.RandomForestClassifier(n_estimators=100, 
                                                         n_jobs = -1)]]
    
    #%%
    # set up all the possible samplers
    samplers = [
        ['ROS', RandomOverSampler()],
        ['SMOTE', SMOTE()],
        ['ADASYN', ADASYN()],
        ['Standard', DummySampler()]
    ]
    
    #%%
    # set up the pipeline: over-sampler then classifier
    #pipelines = [
    #    ['{}-{}'.format(sampler[0], classifier[0]),
    #     make_pipeline(sampler[1], classifier[1])]
    #    for sampler in samplers
    #]
    
    #%% set up the cross-validation split    
    # 3-fold stratified sampling: it returns a list of indices
    nfolds = 3
    cv = StratifiedKFold(n_splits=nfolds)
    # or if I want random and possibly overlapping subset (so not a partition)
    #cv = StratifiedShuffleSplit(n_splits=4, test_size = 1/4)
    
    #%
    classes = ['Pregraduate','Graduate','Postgraduate']
    cv_models = []
    ix = 0
    for train, test in cv.split(x, y):
        results = {}
        results_dummy = {}
        for name, sampler in samplers[:1]:
            for cname, classifier in classifiers:
                fullname= name + '+' + cname
                t0 = time.time()
                clf = deepcopy(classifier)
                #ensemble.RandomForestClassifier(n_estimators=100, n_jobs = -1)
                X_res, y_res = sampler.fit_resample(x[train],y[train])
                clf_descr, score, pred, clf, F10, CM0, CM_norm0 = benchmark(clf, X_res, y_res,
                                                        x[test],y[test], classes)
                # join these predictions to the ones from the MAP and compute
                #y_test_full = np.concatenate((y[test],y_test_MAP))
                #y_pred_full = np.concatenate((pred, y_pred_MAP))
                #F1, CM, CM_norm = benchmark_scores(y_test_full, y_pred_full, classes)
                    
                #probas_ = clf.fit(X_res, y_res).predict_proba(x[test])
                #pipeline.fit(FEATURES[train], TARGETS[train]).predict_proba(
                #        FEATURES[test])
                # add everything to a dictionary
                results[fullname] = {'name':clf_descr, 'accuracy':score,
                           'pred': pred, 'classifier': clf, 
                           'score_partial': {'F1': F10, 'CM': CM0, 'CM_norm': CM_norm0}}
                           #'score_total': {'F1': F1, 'CM': CM, 'CM_norm': CM_norm}, 
                
                print_elapsed(t0, 'training RFC on split number {} and sampler {}'.format(
                        ix,name))
                # for comparison, build a dummy classifier
                clf_dummy = DummyClassifier(strategy='stratified')#, random_state=0)
                clf_descr, score, pred, clf, F1, CM, CM_norm = benchmark(clf_dummy, 
                                        X_res, y_res, x[test],y[test], classes)
                results_dummy[fullname] = {'name':clf_descr, 'accuracy':score,
                           'pred': pred, 'classifier': clf, 
                           'score_partial': {'F1': F10, 'CM': CM0, 'CM_norm': CM_norm0}}
                           #'score_total': {'F1': F1, 'CM': CM, 'CM_norm': CM_norm}, 
        
        # append all to a list
        cv_models.append({'train_indices': train, 'test_indices': test,
                          'classes': classes, 'results': results, 
                          'results_dummy': results_dummy, 'ix_cv': ix})#, 
                          #'y_pred_MAP' : y_pred_MAP, 'cv_index': ix})
        ix += 1
    
        #nb_of_splits = cv.get_n_splits(x, y)
    
    #%
    if SAVERES:
        with open(os.path.join(saveoutput,
            '{}-fold-partition_results_hybrid_rfc.pickle'.format(nfolds)),'wb') as f:
            pickle.dump((model_soc4_to_edu_cat1, model_titles_to_edu_cat2,
                     cv_models),f)
    
    return {'model_soc': model_soc4_to_edu_cat1, 
            'model_title': model_titles_to_edu_cat2,
            'cv_models_partial': cv_models,'SOC_enc': enc_left}

#%%
'''
Note: when I go to evaluate this model, I think I should evaluate it in the
following way:

    1. Label the first portion of the data according to the MAP
    2. Using a partition-style CV label each portion according to the predictions
        obtained when it is used as the test set [I can do this if I keep the 
        indices used in the partition. Note that the test indices are 
        not oversampled]
    3. Compute the F1 score and the CM on the whole dataset
    4. Ideally, I would repeat point 2 for different models and different samplers
        (SMOTE, or another ensemble classifier like ADABOOST)
    5. If I have more than one trained model, choose the one with the best
        accuracy overall - it shouldn't suffer from overfitting, since the 
        labelling of x from the trained model is always done when x was not 
        used directly to train the model.
    6. Also compute the uncertainty score for each x classified by the trained
        model

It is a bit unorthodox, but it kinda corresponds to taking the average of the 
cross-validated F1 score across all partitions
'''


