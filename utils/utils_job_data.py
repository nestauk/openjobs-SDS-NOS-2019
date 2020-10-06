#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:19:02 2019

@author: stefgarasto
"""
'''
Imports
'''
import pandas as pd
import time
import os
import pickle
import numpy as np
from nltk.util import ngrams
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from utils_general import *

#%%
'''
These functions were developed when working with data from Burning Glass.
They might be adapted to work with other job adverts data, but with some necessary caution.
'''

#%%
def extract_skills_list(job):
    '''# split where the comma is.
    # First and last characters in the whole string are the parenthesis
    # First and last characters in each skills are a space and a quote'''
    tmp = [t[1:-1] for t in job[1:-1].split(',')]
    '''# the skills from the second onward have still a quote at the end'''
    tmp = tmp[0:1] + [t[1:] for t in tmp[1:]]
    return tmp

#%%
def group_edu(x):
    ''' Group years of education into discrete categories'''
    if isinstance(x,float) & (not np.isnan(x)):
        if x<16:
            return 'Pregraduate'
        elif x<18:
            return 'Graduate'
        else:
            return 'Postgraduate'
    else:
        return np.nan

#%%
def group_eduv2(x):
    ''' Group years of education into discrete categories (different strategy)'''
    if isinstance(x,float) & (not np.isnan(x)):
        if x<16:
            return 'Pregraduate'
        elif x<20:
            return 'Graduate'
        else:
            return 'Postgraduate'
    else:
        return np.nan

#%%
def group_exp3(x):
    ''' Group years of experience into 3 discrete categories'''
    if isinstance(x,float) & (not np.isnan(x)):
        # note that it's usually in months
        if x<=18:
            return 'Entry-level'
        elif x<=36:
            return 'Mid-level'
        else:
            return 'Senior-level'
    else:
        return np.nan

#%%
def group_exp(x):
    ''' Group years of education into 4 discrete categories'''
    if isinstance(x,float) & (not np.isnan(x)):
        # note that it's usually in months
        if x<=12:
            return 'Entry-level'
        elif x<=24:
            return 'Junior'
        elif x<60:
            return 'Senior'
        else:
            return 'Expert'
    else:
        return np.nan

#%%
def bgsoc_is_newsoc(data):
    ''' Check if we are using SOC code inferred with new algorithm'''
    return np.floor(data['SOC']/10) == data['new_soc']

#%%
def getlondonflag(x):
    ''' Check if job is in London '''
    if isinstance(x,str):
        return x == 'Greater London'
    else:
        return False


#%%
def convert_to_undersc(skill):
    '''
    convert spaces in skill phrases into underscores to use with trained
    w2v model.
    '''
    if len(skill.split(' ')) >1:
        new_i = '-'.join(skill.split(' '))
    else:
        new_i = skill
    return(new_i)

#%%
def convert_from_undersc(skill):
    '''
    convert underscores between terms in skill phrases back to spaces.
    '''
    if len(skill.split('-')) >1:
        new_i = ' '.join(skill.split('-'))
    else:
        new_i = skill
    return(new_i)

#%%
def oov_to_vectors(t, model):
    ''' Function to handle out of vocabulary tokens when creating embeddings'''
    FLAG = 0
    # first try partitioning the wor
    for n in range(len(t)-2,3,-1):
        partition = [t[:n],t[n:]]
        if (partition[0] in model.vocab) and (partition[1] in model.vocab):
            # if both words are in the vocabulary a partition was found
            FLAG = 1
            we_ngrams = (model[partition[0]] + model[partition[1]])/2
            return we_ngrams, FLAG, partition
    if not FLAG:
        # if we didn't find any partition, just get the largest subword that's in
        # the vocabulary (or take an average of all the largest subwords)
        for n in range(len(t)-2,3,-1):
            counter_ngrams = 0
            we_ngrams = np.zeros(model['the'].shape)
            used_grams = []
            for gram in ngrams(t,n):
                gram = ''.join(gram)
                if gram in model.vocab:
                    we_ngrams += model[gram]
                    counter_ngrams += 1
                    FLAG = 1
                    used_grams.append(gram)
            if FLAG:
                # divide by the number of large words used
                we_ngrams = we_ngrams/counter_ngrams
                return we_ngrams, FLAG, used_grams
    # if nothing was found, print the word
    if not FLAG:
        return np.zeros(1), FLAG, np.zeros(1)

#%%
def oov_to_vectors_ngrams(t,model):
    '''# compute all ngrams and take an average of those that exist'''
    # ideally I might want to remove some outliers #TODO
    counter_ngrams = 0
    we_ngrams = np.zeros(model['the'].shape)
    good_grams = []
    for n in range(3,len(t)-2):
        for gram in ngrams(t,n):
            gram = ''.join(gram)
            if gram in model.vocab:
                we_ngrams += model[gram]
                counter_ngrams +=1
                good_grams.append(gram)
    if counter_ngrams>0:
        # if any of the ngrams was found, normalise by the number of
        # ngrams used
        we_ngrams = we_ngrams/counter_ngrams
    # the counter acts as a flag
    return we_ngrams, counter_ngrams


#%%
def sentence_to_vectors(x,model,missing_jt_file0,recovered_jt_file0):
    '''# compute a word embedding for a sentence as the average over the words
    # that make up the job title
    # keep track of how many words don't exist in the vocabulary'''
    counter_oov = 0
    # first check whether the whole word is in the model
    y = convert_to_undersc(x)
    if y in model.vocab:
        return model[y], 1
    # if not, reconvert to discrete words and take the average of them
    y = convert_from_undersc(y)
    # split the sentence into words after removing hyphens: THERE IS NONE
    y = y.split() #y.replace('-',' ').split()
    # remove extra spaces, genitives "'s"
    y = [t.strip().lower().replace('\'s','') for t in y]

    #y = [t.strip().lower().replace('\'s','').split('-') for t in y]
    #y = [t for sublist in y for t in sublist]
    # initialise word embedding and we counter
    we = np.zeros(model['the'].shape, dtype = np.float32)
    we_counter = 0
    missed = []
    for t in y:
        if t in model.vocab:
            we += model[t]
            we_counter += 1
        else:
            we_tmp, flag_oov, recovered_words = oov_to_vectors(t, model)
            if flag_oov:
                we += we_tmp
                we_counter += 1
                with open(recovered_jt_file0,'a') as f:
                    print(recovered_words, file = f)
            else:
                missed.append(t)
                counter_oov += 1
    if counter_oov>0:
        with open(missing_jt_file0,'a') as f:
            print(missed, file = f)
    # normalise by the number of embeddings
    if we_counter>0:
        we = we/we_counter
    return we, we_counter

#%%
def jt_to_vectors(x, model, missing_jt_file0, recovered_jt_file0):
    '''# transform a job title into a word embedding'''
    #missing_jt_file0 = os.path.join(saveoutput,
    #                'job_titles_not_in_dictionary_{}.txt'.format(WHICH_GLOVE))
    #recovered_jt_file0 = os.path.join(saveoutput,
    #            'job_titles_not_in_dictionary_{}_recovered.txt'.format(WHICH_GLOVE))
    we, we_counter = sentence_to_vectors(x,model,
                                missing_jt_file0,recovered_jt_file0)
    if we_counter>0:
        return np.float32(we)
    else:
        # if nothing has been turned to word embedding then this is 0
        return np.zeros(model['the'].shape, dtype = np.float32)

#%%
def skills_to_vectors(x, model, missing_jt_file0, recovered_jt_file0): #WHICH_GLOVE):
    '''# compute a word embedding for the list of skills as an average of averages'''
    if isinstance(x, str):
        skills = eval(x)
    else:
        skills = eval(x.values[0])
    we= np.zeros(model['the'].shape, dtype = np.float32)
    we_counter = 0
    for z in skills:
        we_skill, skill_counter = sentence_to_vectors(z, model,
                                        missing_jt_file0,recovered_jt_file0)
        # note that we_skill is already normalised by skill_counter
        if skill_counter>0:
            # if at least one word making up this skill was in the vocabulary,
            # add it to the overall word embedding:
            we += we_skill
            #jt_to_vectors(z, model, missing_jt_file0, recovered_jt_file0)
            we_counter += 1
    if we_counter>0:
        we = we/we_counter
    return np.float32(we)

#%%
def skills_to_vectors_full_nofile(x, model): #WHICH_GLOVE):
    '''# compute a word embedding for the list of skills as an average of averages'''
    if isinstance(x, str):
        skills = eval(x)
    else:
        skills = eval(x.values[0])
    we= np.zeros((len(skills),model['the'].shape[0]), dtype = np.float32)
    we_counter = 0
    for z in skills:
        we_skill, skill_counter = sentence_to_vectors_nofile(z, model)
        # note that we_skill is already normalised by skill_counter
        if skill_counter>0:
            # if at least one word making up this skill was in the vocabulary,
            # add it to the overall word embedding:
            we[we_counter] = we_skill
            #jt_to_vectors(z, model, missing_jt_file0, recovered_jt_file0)
            we_counter += 1
    if we_counter>0:
        we = we[:we_counter] #/we_counter
    else:
        we = we[:1]
    return np.float32(we)

#%%
def get_all_features(data, enc, enc_london):
    '''# get the one-hot encoding of the SOC codes'''
    X = enc.transform(data['SOC'].values.reshape(-1,1))
    # get the London dummy variable
    X_london = enc_london.transform(data['SOC'].values.reshape(-1,1))
    # get the embedded titles
    feat1 = np.stack(data['title_embedding'].values)
    # get the embedded skills
    feat2 = np.stack(data['skills_embedding'].values)
    # get the mean salary
    feat3 = data['MeanSalary'].values.reshape(-1,1)
    # concatenate all
    return np.concatenate((X.todense(),X_london.todense(),
                           feat1, feat2, feat3), axis=1)

#%% Benchmark classifiers
def benchmark(clf, X_train, y_train, X_test, y_test, classes,
              print_report = False, print_cm = False):
    ''' Evaluate accuracy of random forest classifier'''
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time.time()
    pred = clf.predict(X_test)
    if isinstance(pred[0],(np.float64, float)):
        pred = np.float32(pred)
    pred_proba = np.float32(clf.predict_proba(X_test))
    test_time = time.time() - t0
    print("test time:  %0.3fs" % test_time)

    score = accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    F1 = f1_score(y_test, pred, labels=classes, average='weighted')
    # confusion matrix
    CM = confusion_matrix(y_test, pred, labels=classes)
    CM_norm = CM.astype('float') / CM.sum(axis=1)[:, np.newaxis]

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print()

    if print_report:
        print("classification report:")
        print(classification_report(y_test, pred))

    if print_cm:
        print("confusion matrix:")
        print(CM)

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, pred, pred_proba, clf, F1, CM, CM_norm# train_time, test_time

#%%
def benchmark_scores(y_test, y_pred, classes):
    '''compute the classification scores given targets and predictions'''
    F1 = f1_score(y_test, y_pred, labels=classes, average='weighted')
    # confusion matrix
    CM = confusion_matrix(y_test, y_pred, labels=classes)
    CM_norm = CM.astype('float') / CM.sum(axis=1)[:, np.newaxis]
    return F1, CM, CM_norm

#%%
def balanced_sample(data,M = 0, targetcol = 'target'):
    '''To sample from the data with equalised classes'''
    indices = data.index
    postgrad = indices[data[targetcol]=='Postgraduate']
    if M == 0:
        M = len(postgrad)
    postgrad = postgrad[np.random.choice(np.arange(len(postgrad)), size = M)]

    grad = indices[data[targetcol]=='Graduate']
    grad = grad[np.random.choice(np.arange(len(grad)), size = M)]

    pregrad= indices[data['target']=='Pregraduate']
    pregrad = pregrad[np.random.choice(np.arange(len(pregrad)),
                                       size =M)]
    sample = pd.concat([data.loc[postgrad],data.loc[pregrad],data.loc[grad]])
    return sample

#%%
def prep_for_gensim(list_of_terms, some_model, weights = None):
    ''' change tokens to a format to use with gensim (unigrams)'''
    # replace space with underscore
    new_terms = [convert_to_undersc(elem) for elem in list_of_terms]
    # check if each element in the list is in the model
    is_in = [elem for elem in new_terms if elem in some_model]
    # also check the weights
    if weights:
        weights_in = [weights[ix] for ix,elem in enumerate(new_terms)
                        if elem in some_model]
    # only return the element in the model
    return is_in, weights_in

#%%
def prep_for_gensim_bigrams(list_of_terms, some_model, weights = None):
    ''' change tokens to a format to use with gensim (bigrams)'''
    # replace space with underscore
    #new_terms = [convert_to_undersc(elem) for elem in list_of_terms]
    # check if each element in the list is in the model
    all_indices = np.arange(len(list_of_terms))
    indices_bool = [all([t in some_model for t in elem.split()])
        for elem in list_of_terms]
    indices_bool_undersc = [convert_to_undersc(elem) in some_model for elem in list_of_terms]
    #indices_in = [t for ix,t in enumerate(all_indices) if indices_bool[ix] |
    #        indices_bool_undersc[ix]]
    #indices_in = [ix for ix,elem in enumerate(list_of_terms) if
    #                         all([t in some_model for t in elem.split()])]
    #indices_in_undersc = [ix for ix,elem in enumerate(list_of_terms) if
    #                      convert_to_undersc(elem) in model]
    #is_in = [elem for ix,elem in enumerate(list_of_terms) if ix in indices_in]
    is_in = [elem for ix,elem in enumerate(list_of_terms) if
             indices_bool[ix] | indices_bool_undersc[ix]]
    # also check the weights
    if weights:
        weights_in = [weights[ix] for ix,elem in enumerate(list_of_terms)
                        if indices_bool[ix] | indices_bool_undersc[ix]]#ix in indices_in]
    else:
        weights_in = None
    # only return the element in the model
    return is_in, weights_in

#%%
def convert_glove_model(glove_name, dims = ['50','100','200']):
    ''' Convert glove model to use with Gensim'''
    from gensim.scripts.glove2word2vec import glove2word2vec

    glove_dir = '/Users/stefgarasto/Local-Data/wordvecs/glove.twitter.27B'

    # to make the glove model file compatible with gensim
    for dim in dims:
        glove_file = os.path.join(glove_dir,'{}.{}d.txt'.format(glove_name,dim))
        tmp_file = os.path.join(glove_dir, 'word2vec.{}.{}d.txt'.format(
                glove_name,dim) )
        _ = glove2word2vec(glove_file, tmp_file)
    return 'Done'


#%%
def sentence_to_vectors_nofile(x,model_l):
    '''# compute a word embedding for a sentence as the average over the words
    # that make up a text sentence'''
    counter_oov = 0
    # first check whether the whole word is in the model
    y = convert_to_undersc(x)
    if y in model_l.vocab:
        return model_l[y], 1
    # if not, reconvert to discrete words and take the average of them
    y = convert_from_undersc(y)
    # split the sentence into words after removing hyphens: THERE IS NONE
    y = y.split()
    # remove extra spaces, genitives "'s"
    y = [t.strip().lower().replace('\'s','') for t in y]

    # initialise word embedding and we counter
    we = np.zeros(model_l['the'].shape, dtype = np.float32)
    we_counter = 0
    missed = []
    for t in y:
        if t in model_l.vocab:
            we += model_l[t]
            we_counter += 1
        else:
            we_tmp, flag_oov, recovered_words = oov_to_vectors(t, model_l)
            if flag_oov:
                we += we_tmp
                we_counter += 1
            else:
                missed.append(t)
                counter_oov += 1
    # normalise by the number of embeddings
    if we_counter>0:
        we = we/we_counter
    return we, we_counter

#%%
def jt_to_vectors_nofile(x, model):
    '''# transform a job title into a word embedding'''
    we, we_counter = sentence_to_vectors_nofile(x,model)
    if we_counter>0:
        return np.float32(we)
    else:
        # if nothing has been turned to word embedding then this is 0
        return np.zeros(model['the'].shape, dtype = np.float32)

#%%
def skills_to_vectors_nofile(x, model):
    '''# compute a word embedding for the list of skills as an average of averages'''
    if isinstance(x, str):
        skills = eval(x)
    else:
        skills = eval(x.values[0])
    we= np.zeros(model['the'].shape, dtype = np.float32)
    we_counter = 0
    for z in skills:
        we_skill, skill_counter = sentence_to_vectors_nofile(z, model)
        # note that we_skill is already normalised by skill_counter
        if skill_counter>0:
            # if at least one word making up this skill was in the vocabulary,
            # add it to the overall word embedding:
            we += we_skill
            we_counter += 1
    if we_counter>0:
        we = we/we_counter
    return np.float32(we)


#%% extra SOCs I need to match from NOS
matches_oobsoc_to_soc = {'3133': '3113', '8140': '8149','1170':'1173','9': '1221'}
matches_oobsoc_to_soc2 = {3133.0: 3113.0, 8140.0: 8149.0,
                          1170.0: 1173.0, 9: 1221.0}


#%%
#
def plotHM(hm2plot, xlabel = '', ylabel = '', title = 'Heatmap',
           xindices = None, yindices= None, normalize = False, w=6, h=6):
    '''Plot annotated heatmap'''

    fig, ax = plt.subplots(figsize = (w,h))
    if isinstance(hm2plot, np.ndarray):
        hm2plot = pd.DataFrame(hm2plot, index = yindices, columns = xindices)
    im = sns.heatmap(hm2plot, cmap=sns.cm.rocket_r)
    #ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    #im.set(xticks=np.arange(hm2plot.values.shape[1]),
    #       yticks=np.arange(hm2plot.values.shape[0]),
    #       # ... and label them with the respective list entries
    plt.title(title)
    plt.ylabel(ylabel, fontsize = 18, rotation = 'horizontal')
    plt.xlabel(xlabel, fontsize = 18)

    # Rotate the tick labels and set their alignment.
    plt.setp(im.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if isinstance(hm2plot.values[0,0],float) else 'd'
    thresh = hm2plot.values.max() / 2.
    for i in range(hm2plot.values.shape[0]):
        for j in range(hm2plot.values.shape[1]):
            ax.text(j+.5, i+.5, format(hm2plot.values[i, j], fmt),
                    ha="center", va="center", fontsize = 14,
                    color="white" if hm2plot.values[i, j] > thresh else "black")
    fig.tight_layout()
    return fig, im
