#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 18:02:03 2019

@author: stefgarasto

This scripts builds the "hybrid model" whereby some educational requirements
are estimated by MAP on the SOC codes (most common classification per SOC code,
when the MAP is > a certain threshold); some by MAP on common job titles; some
by random forest classification.

Everything goes through cross validation

It assumes that bg_load_and_prepare_data has been run
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
from copy import deepcopy
#from imblearn.ensemble import BalancedRandomForestClassifier
import time
from sklearn.metrics import precision_recall_fscore_support

#%% Benchmark classifiers
from utils_bg import saveoutput, socnames_dict, print_elapsed, get_all_features
from utils_bg import benchmark, benchmark_scores

#%%
def train_full_hybrid(bgdatasmall, enc_london, classifiers = None,
                     SAVERES = True, SAVEFIG = False, match_th = .9, nfolds = 4,
                     extra_args = {'SUPER': 'all', 'CV': 'StratifiedKFold',
                                  'CVshuffle': False, 'CVrs': 42},
                    WHICH_GLOVE = 'glove.6B.100d', target_var = 'Edu'):
    
    #%
    #LW = 2
    #RANDOM_STATE = 42
    
    class DummySampler:
    
        def sample(self, X, y):
            return X, y
    
        def fit(self, X, y):
            return self
    
        def fit_resample(self, X, y):
            return self.sample(X, y)
    
    #%%
    print('Setting up the classifier')
    # set up the classifiers
    if not classifiers:
        classifiers = [['RFC', ensemble.RandomForestClassifier(n_estimators=100, 
                                                         n_jobs = -1)],
                    ['CalRFC', CalibratedClassifierCV(ensemble.RandomForestClassifier(
                            n_estimators=100, n_jobs = -1), 
                        cv= StratifiedKFold(n_splits=2), method = 'isotonic')]]
    
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
    #    ['{}-{}'.format(sampler[0], classifiers[0]),
    #     make_pipeline(sampler[1], classifiers[1])]
    #    for sampler in samplers
    #]
        
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
    - split into training and testing. OK
    - classify easy SOCs. OK
    - classify easy titles. OK
    - remove classified socs and titles from training data. OK
    - remove classified socs and titles from test data, but keep their location
        (or keep the corresponding test targets separate as well). OK
    - get predictions on easy-to-classify test data
    - fit SOC encoder on training data
    - oversample training data
    - fit classifier on training data
    - get predictions on "leftover" test data
    - concatenate targets and predictions in a consistent way
    - get F1, accuracy and CM
    - save results in a list
    
    '''
    #%%
    # 3-fold stratified sampling: it returns a list of indices
    #nfolds = 4
    print('Setting up the cross-validation iterator')
    if extra_args['CV'] == 'StratifiedKFold':
        cv = StratifiedKFold(n_splits=nfolds, shuffle =extra_args['CVshuffle'], 
                         random_state = extra_args['CVrs'])
    else:
        raise ValueError
    # or if I want random and possibly overlapping subset (so not a partition)
    #cv = StratifiedShuffleSplit(n_splits=nfolds, test_size = 1/nfolds)
    
    #%%
    cv_models = []
    ix_cv = 1
    print('Starting the cross-validation loop to train the models')
    for train, test in cv.split(bgdatasmall['SOC'], bgdatasmall[target_var]):
        t0 = time.time()
        '''
        Split into training and testing
        '''
        bgdata_train = bgdatasmall.iloc[train]
        bgdata_test = bgdatasmall.iloc[test]
        ''' 
        Compute the posterior of Edu given SOC codes.
        '''
        # get frequency table
        joint_dist = pd.crosstab(bgdata_train[target_var], 
                            bgdata_train['SOC'])
        # the baseline accuracy of any model is the proportion of the biggest class
        #baseline_accuracy = np.around((joint_dist.sum())/(
        #        joint_dist.sum().sum()).max())
        
        # normalise for each SOC code
        posterior = joint_dist/joint_dist.sum()
        
        # normalise by counts of MinEdu
        likelihood = (joint_dist.T)/(joint_dist.T.sum())
        likelihood = (likelihood.T)
        # this should correspond to the MultinomionalNB with alpha=0, fit_prior = False
        #  so that likelihood == np.exp(clf_basic.feature_log_prob_)
        
        
        # now get the prior
        prior = bgdata_train[target_var].value_counts()
        prior = (prior/prior.sum()).values
        
        #%
        ''' Collect and print to file all the SOC that are matched to a category more 
        than 90% of the time. Store the results in model_soc4_to_edu_cat1
        '''
        matched_socs = posterior.columns[posterior.max()>match_th]
                #(posterior.max()>match_th) & (joint_dist.sum()>1)]
        model_soc4_to_edu_cat1 = {}
        ja_tot = 0
        for isoc in matched_socs:
            matched_to = posterior.index[posterior[isoc]>match_th].values[0]
            model_soc4_to_edu_cat1[isoc] = {'match': matched_to, 
                        'known accuracy': posterior.loc[matched_to][isoc]}
        
        #%
        '''
        for each SOC that has not been matched get the 10 most common job titles and
        classify them as above. Store the results in model_titles_to_edu_cat2
        '''
        soc_by_jt = pd.crosstab(bgdata_train['title_processed'], bgdata_train['SOC'])
        jt_nb_by_socs = (soc_by_jt>0).sum()
        groups = bgdata_train.groupby('title_processed')
        non_matched_socs = list(set(jt_nb_by_socs.index) - set(matched_socs))
        model_titles_to_edu_cat2 = {}
        counter =0
        ja_tot = 0
        if SAVERES and False:
            file2use = ''.join(['titles_matched_to_{}_category2_'.format(
                                target_var.lower()),
                                'cv{}-{}_{:.0f}_20190729.txt'.format(
                                 ix_cv,nfolds,100*match_th)])
            with open(os.path.join(saveoutput,file2use), 'w') as f:
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
                            print('\n',file=f)
                            print(
                            ''.join(['\'{}\' was matched to \'{}\' {:.2f}% '.format(
                                        ijt, 
                                        matched_to,
                                        100*edu_cat_dist.loc[matched_to]),
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
        jobs_classified_train = bgdata_train['SOC'].isin(list(model_soc4_to_edu_cat1.keys(
                ))) | bgdata_train['title_processed'].isin(
                                    list(model_titles_to_edu_cat2.keys()))
        bgleft_train = bgdata_train[~jobs_classified_train]
        # test data
        jobs_classified_test = bgdata_test['SOC'].isin(list(model_soc4_to_edu_cat1.keys(
                ))) | bgdata_test['title_processed'].isin(
                                    list(model_titles_to_edu_cat2.keys()))
        bgleft_test = bgdata_test[~jobs_classified_test]
        
        
        '''
        Make predictions on part of the test dataset based on the SOC and titles models
        '''
        # initialise the prediction column in the test dataset
        bgdata_test['Target_pred'] = 'Unknown'
        # assign each known soc code to its match
        for isoc in model_soc4_to_edu_cat1:
            locations = bgdata_test['SOC']==isoc
            bgdata_test['Target_pred'][locations] = model_soc4_to_edu_cat1[isoc]['match']
        # assign each known job title to its match
        for isoc in model_titles_to_edu_cat2:
            locations = bgdata_test['title_processed']==isoc
            bgdata_test['Target_pred'][locations] = model_titles_to_edu_cat2[isoc]['match']
        # at the end of this all the rows with ones in jobs_classified_test should
        # have been assigned
        y_pred_MAP = bgdata_test['Target_pred'][jobs_classified_test].values
        y_test_MAP = bgdata_test[target_var][jobs_classified_test].values
        
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
        x_test = get_all_features(bgleft_test, enc_left, enc_london)
        #xunit = StandardScaler().fit_transform(x)
        print_elapsed(t0,'task above')
        
        # add scaling? 
        #% get the classification targets
        y_train = bgleft_train[target_var].values
        y_test = bgleft_test[target_var].values
        
        print(np.unique(y_test),np.unique(y_train))
        
        #%%
        results = {}
        results_dummy = {}
        for name, sampler in samplers[:1]:
            for cname, classifier in classifiers:
                fullname = name + '+' + cname
                t0 = time.time()
                clf = deepcopy(classifier)
                #ensemble.RandomForestClassifier(n_estimators=100, n_jobs = -1)
                X_res, y_res = sampler.fit_resample(x_train,y_train)
                X_test_res, y_test_res = sampler.fit_resample(x_test, y_test)
                clf_descr, score, pred, pred_probs, clf, F10, CM0, CM_norm0 = benchmark(
                                                    clf, X_res, y_res,
                                                    x_test,y_test, classes)
                # get all measures (precision, recalls, f-beta score and support)
                prfs0 = precision_recall_fscore_support(y_test, pred, labels = classes)
                
                # join these predictions to the ones from the MAP and compute
                y_test_full = np.concatenate((y_test,y_test_MAP))
                y_pred_full = np.concatenate((pred, y_pred_MAP))
                F1, CM, CM_norm = benchmark_scores(y_test_full, y_pred_full, classes)
                # get all measures (precision, recalls, f-beta score and support)
                prfs = precision_recall_fscore_support(y_test_full, y_pred_full, 
                                                            labels = classes)
                
                # now make predictions and probabilities for the balanced test sets
                pred_blc = clf.predict(X_test_res)
                pred_probs_blc = np.float32(clf.predict_proba(X_test_res))
                # get accuracy scores for these
                F1_blc, CM_blc, CM_norm_blc = benchmark_scores(y_test_res, pred_blc, 
                                                               classes)
                # get all measures (precision, recalls, f-beta score and support)
                prfs_blc = precision_recall_fscore_support(y_test_res, pred_blc, 
                                                           labels = classes)
                
                # add everything to the dictionary
                results[fullname] = {'name':clf_descr, 'accuracy':score,
                       'pred': pred,   'pred_probs': pred_probs,
                       'pred_blc': pred_blc, 'pred_probs_blc': pred_probs_blc,
                       'score_total': {'F1': F1, 'CM': CM, 'CM_norm': CM_norm,
                                       'prfs' : prfs}, 
                       'score_partial': {'F1': F10, 'CM': CM0, 'CM_norm': CM_norm0,
                                         'prfs': prfs0},
                       'score_partial_blc': {'F1': F1_blc, 'CM': CM_blc, 
                                    'CM_norm': CM_norm_blc, 'prfs': prfs_blc}}
                # not saving anymore bc of size: 'classifier': clf, 
                
                print_elapsed(t0, 'training RFC on split number {} and sampler {}'.format(
                        ix_cv, name))
                
                ## for comparison, build a dummy classifier
                #clf_dummy = DummyClassifier(strategy='stratified')#, random_state=0)
                #clf_descr, score, pred, pred_probs, clf, F10, CM0, CM_norm0 = benchmark(
                #                        clf_dummy, X_res, y_res,
                #                        x_test, y_test, classes)
                #
                # join these predictions to the ones from the MAP and compute
                #y_test_full = np.concatenate((y_test,y_test_MAP))
                #y_pred_full = np.concatenate((pred, y_pred_MAP))
                #F1, CM, CM_norm = benchmark_scores(y_test_full, y_pred_full, classes)
                #
                ## add everything to a dictionary
                #results_dummy[fullname] = {'name':clf_descr, 'accuracy':score,
                #       'pred': pred,  'pred_probs': pred_probs,
                #       'score_total': {'F1': F1, 'CM': CM, 'CM_norm': CM_norm}, 
                #       'score_partial': {'F1': F10, 'CM': CM0, 'CM_norm': CM_norm0}}
                # not saving anymore: , 
    
        #%% append the result to a list (one element for each partition)
        cv_models.append({'model_soc': model_soc4_to_edu_cat1, 
                          'model_title': model_titles_to_edu_cat2, 
                          'classes': classes, 'results': results, 
                          'cv_index': ix_cv,
                          'jobs_classified_train': jobs_classified_train, 
                          'jobs_classified_test': jobs_classified_test,
                          'SOC_enc': enc_left, 'extra_args': extra_args,
                          'train_indices': train, 'test_indices': test})
            # not saving these anymore: 'train_indices': train, 'test_indices': test
            # 'results_dummy': results_dummy, 'y_pred_MAP' : y_pred_MAP, 
        ix_cv+=1
    #%%
    if SAVERES:
        with open(os.path.join(saveoutput,
                 '{}-fold-partition_{:.0f}_results_hybrid_rfc_fullcv_{}_{}.pickle'.format(
                 nfolds,match_th*100,WHICH_GLOVE,target_var)),'wb') as f:
            pickle.dump(cv_models,f)
        
    #%%
    '''
    Note: when I go to evaluate this model, I think I should evaluate it in the
    following way:
    
        1. Using a partition-style CV label each portion according to the predictions
            obtained when it is used as the test set [I can do this if I keep the 
            indices used in the partition. Note that the test indices are 
            not oversampled]. This applies to both the MAP and the trained model.
        2. Compute the F1 score and the CM on the whole dataset
        3. Ideally, I would repeat point 1 for different models and different samplers
            (SMOTE, or another ensemble classifier like ADABOOST)
        4. If I have more than one trained model, choose the one with the best
            accuracy overall - it shouldn't suffer from overfitting, since the 
            labelling of x from the trained model is always done when x was not 
            used directly to train the model.
        5. Also compute the uncertainty score for each x classified by the trained
            model
    
    It is a bit unorthodox, but it kinda corresponds to taking the average of the 
    cross-validated F1 score across all partitions
    '''
    return cv_models


if __name__ == "__main__":
    cv_models = train_full_hybrid(bgdatasmall, enc_london, target_var = target_var)