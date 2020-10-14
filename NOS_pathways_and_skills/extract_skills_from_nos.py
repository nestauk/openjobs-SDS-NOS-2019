#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:37:38 2019

@author: stefgarasto

This script is to detect 
"""

import pandas as pd
import time
import pickle
from fuzzywuzzy import process, fuzz
import multiprocessing
from functools import partial
from contextlib import contextmanager
from utils_skills_clusters import bottom_layer, skills_ext, df_match_final
from utils_skills_clusters import skills_ext_long, nesta_skills
from utils_nlp import tidy_desc_with_pos, tidy_desc_with_pos_skills
from copy import deepcopy

#%%
# load the skills matched to a cluster
# save the final dataframe
matches_dir = '/Users/stefgarasto/Google Drive/Documents/scripts/NOS/results/nos_vs_taxonomy/'
with open(matches_dir + '/final_matches_nesta_vs_ext.pickle','rb') as f:
    skills_matches_consensus_ext = pickle.load(f)

with open(matches_dir + '/final_matches_nesta_vs_ext_emsi.pickle','rb') as f:
    skills_matches_consensus_emsi = pickle.load(f)

skills_matches_consensus = skills_matches_consensus_ext.append(
        skills_matches_consensus_emsi)

# remove duplicated skills
skills_matches_consensus2 = skills_matches_consensus.reset_index().drop_duplicates().set_index('index')

# create the lemmatised version
skills_matches_consensus2['lemmatised'] = skills_matches_consensus2.index.map(lambda x: 
    ' '.join(tidy_desc_with_pos_skills(x,'all')))
skills_matches_consensus2['lemmatised'] = skills_matches_consensus2['lemmatised'].map(
        lambda x : x.replace('\'',''))

# remove duplicates based on the lemmatised version - this will ignore the index and
# check whether there is any row with same cluster and lemmatise version
skills_matches_consensus2 = skills_matches_consensus2.drop_duplicates()

#%% put lemmatised skills in their own dataframe
#new_skills_lemmas = skills_matches_consensus2['lemmatised'].map
skills_lemmas = skills_matches_consensus2.reset_index().drop('original_skill', axis=1).rename(
        columns={'index': 'original_skill'}).rename(
        columns={'lemmatised':'index'})[['consensus_cluster','index','original_skill']]
skills_lemmas = skills_lemmas.set_index('index')

# avoid duplication
skills_lemmas = skills_lemmas[skills_lemmas.index.map(lambda x: x not in 
                                            skills_matches_consensus2.index)]
# remove lemmas with length less than 3
skills_lemmas = skills_lemmas[skills_lemmas.index.map(lambda x: len(x)>=3)]

# remove short and ambiguous lemmas - assuming they're acronyms, I don't need to lemmatise them
# only keep some selected ones
good_short_lemmas = ['law','act','ssa','age','ibm','dye','iso']
skills_lemmas = skills_lemmas[(skills_lemmas.index.map(len)>3) | (
        skills_lemmas.index.map(lambda x: x in good_short_lemmas).values)]

ambiguous_lemmas = ['then','view','link','user','act','factor','type','potential',
                    'log','sheet','base','weather','section','safe','level','basic',
                    'target','power','analyse','security','encoder','strategy',
                    'support','estimate','sigma','strut','label','scale',
                    'alpha','meal','index','model','notepad','test','bone',
                    'bread','curry','dice','dime','grilling','mike','neve',
                    'quartz','skin','surface','data','target','thin',
                    'ultra','wing','client','secure','adapt']
# socket --> it support / networks
# remember surface from surfacing!!

# then comes from if this then that
# view comes from ng view
# link comes from link 4
# user comes from user32
# act comes from acting but might mean something else
# factor comes frmo 12factor
# type I'm not sure, but maybe typing - this is tricky because it'll be lemmatised in
    # NOS but it could mean "type" or "typing" with no way of discerning
# potential maybe from zeta potential
# log maybe from logly or logging - again, tricky one a bit
# sheet maybe from one sheet
# base probably from base24
# weather comes from weathering (which is an erosion process)
# section comes from section508
# safe comes from safes (mainframe programming)
# level comes from levelling - it now refers to something quite different
# basic comes from basic a+ which again means something else
# data comes from 1010data, which is something else
# similarly target comes from target 3001!, which is something else
# power comes from power3
# scale comes from scaling
    
# not added one:
# sort from sorting
# panel from paneling

skills_lemmas = skills_lemmas[skills_lemmas.index.map(lambda x: 
    x not in ambiguous_lemmas)]

# remove duplicates
skills_lemmas = skills_lemmas[
            ~skills_lemmas.index.duplicated(keep='last')]

# append to the main dataframe
skills_matches_consensus2['original_skill'] = skills_matches_consensus2.index
skills_matches_consensus4 = skills_matches_consensus2.append(skills_lemmas,
                                                             sort=False)
skills_matches_consensus2 = deepcopy(skills_matches_consensus4)
#%% do some more small changes, just to check if anything changes
'''
external link?, flux, gps time, 
quotations, recipes, reservations, patents,
'''
skills_matches_consensus4.loc['food']['consensus_cluster'] = 'retail'
skills_matches_consensus4.loc['foods']['consensus_cluster'] = 'retail'
skills_matches_consensus4.loc['fault']['consensus_cluster'] = 'electrical engineering'
skills_matches_consensus4.loc['history']['consensus_cluster'] = 'teaching'
skills_matches_consensus4.loc['private pub']['consensus_cluster'] = 'retail'
skills_matches_consensus4.loc['hormone']['consensus_cluster'] = 'molecular biology of cancer'
skills_matches_consensus4.loc['hormones']['consensus_cluster'] = 'molecular biology of cancer'
skills_matches_consensus4.loc['judiciary']['consensus_cluster'] = 'legal services'
skills_matches_consensus4.loc['standard deviation']['consensus_cluster'] = 'research methods and statistics'
skills_matches_consensus4.loc['statistical classification']['consensus_cluster'] = 'research methods and statistics'
skills_matches_consensus4.loc['coats']['consensus_cluster'] = 'construction'
skills_matches_consensus4.loc['coat']['consensus_cluster'] = 'construction'
skills_matches_consensus4.loc['seafood']['consensus_cluster'] = 'retail'
skills_matches_consensus4.loc['tires']['consensus_cluster'] = 'driving and automotive maintenance'
skills_matches_consensus4.loc['tire']['consensus_cluster'] = 'driving and automotive maintenance'
skills_matches_consensus4.loc['hair']['consensus_cluster'] = 'physiotherapy and beauty'
skills_matches_consensus4.loc['quotations']['consensus_cluster'] = 'retail management'
skills_matches_consensus4.loc['quotation']['consensus_cluster'] = 'retail management'
#skills_matches_consensus4.loc['external links']['consensus_cluster'] = 'web development'
#skills_matches_consensus4.loc['external link']['consensus_cluster'] = 'web development'
skills_matches_consensus4.loc['motorsport']['consensus_cluster'] = 'extracurricular activities and childcare'

#%% do some more changes for "low vision support" cluster
skills_to_change = ['active listening',
 'agility',
 'electronic signature','electronic signatures',
 'hand pump','hand pumps',
 'headset', 'headsets',
 'heat gun', 'heat guns',
 'motivate others',
 'use microphone',
 'interact with others','interact others',
 'hand signal','hand signals']
new_clusters = ['recruitment',
                'physiotherapy and beauty',
                'office administration','office administration',
                'heating, ventilation and plumbing','heating, ventilation and plumbing',
                'multimedia production','multimedia production',
                'construction','construction',
                'general sales',
                'multimedia production',
                'employee development','employee development',
                'construction','construction']
for ix,t in enumerate(skills_to_change):
    skills_matches_consensus4.loc[t]['consensus_cluster'] = new_clusters[ix]

#%%
# lemmatise all skills using 
#%%
@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()
   
#%%
def exact_skills_match_in_text(row_nos, skills_to_match = ['non_existent_string']): #which_tax = 'nesta'):
    text_nos = row_nos['clean_full_text']
    return [t for t in skills_to_match if ' '+t+' ' in text_nos]

def exact_skills_match_token(row_nos, skills_to_match = ['non_existent_string']):
    text_nos = row_nos['clean_tokens']
    pass
        
#%
def extract_exact_matches(df_nos_local, which_tax = 'nesta'):
    # first check for exact matches (of the type "is SKILL 'IN' TEXT?")
    assert(which_tax in ['nesta','emsi','external'])
    
    if which_tax in ['nesta','emsi']:
        skills_to_match = list(df_match_final[which_tax])
    else:
        skills_to_match = list(skills_matches_consensus4.index)
        #skills_to_match += [t for t in 
        #                    list(skills_matches_consensus3['lemmatised']) if
        #                    len(t)>=3]
        # just in case there are still duplicates        
        skills_to_match = list(set(skills_to_match))
    t0 = time.time()
    with poolcontext(processes=4) as pool:
        all_exact_matches = []
        for istart in range(0,len(df_nos_local),100):
            print('Matching skills exactly. Got to iteration ', istart)
            all_exact_matches += pool.map(partial(exact_skills_match_in_text, 
                                                  skills_to_match= skills_to_match), 
                                     [df_nos_local.iloc[ix] for ix in 
                                              range(istart,min(istart+100, len(df_nos_local)))])
    print('Total time elapsed (in seconds): ', time.time()-t0)
    return all_exact_matches


EXECUTE = False
if EXECUTE:
    all_exact_matches_nesta = extract_exact_matches(df_nos2, 'nesta')
    all_exact_matches_emsi = extract_exact_matches(df_nos2, 'emsi')

#%%
cutoff_by_length = {
        1: 100, 2:100, 3:100, 4:100, 5:100}
for ix in range(6,11):
    cutoff_by_length[ix] = 95
for ix in range(11,21):
    cutoff_by_length[ix] = 86
for ix in range(21,36):
    cutoff_by_length[ix] = 76
for ix in range(36,55):
    cutoff_by_length[ix] = 65

#%%
def fuzzy_skills_match_in_text(row_nos, skills_to_match = ['non_existent_string']): 
    #which_tax = 'nesta'):
    text_nos = row_nos['clean_full_text']
    results= process.extractBests(text_nos, skills_to_match, limit=50, score_cutoff=50,
                                scorer = fuzz.partial_ratio) #partial_token_sort_ratio)
    return [t[0] for t in results if t[1]>=cutoff_by_length[len(t[0])]]

#%%
def extract_fuzzy_matches(df_nos_local, which_tax = 'nesta'):
    # first check for exact matches (of the type "is SKILL 'IN' TEXT?")
    assert(which_tax in ['nesta','emsi','external'])
    
    if which_tax in ['nesta','emsi']:
        skills_to_match = list(df_match_final[which_tax])
    else:
        skills_to_match = list(skills_matches_consensus4.index)
    t0 = time.time()
    with poolcontext(processes=4) as pool:
        all_fuzzy_matches = []
        for istart in range(0,len(df_nos_local),100):
            print('Matching skills fuzzily. Got to iteration ', istart)
            all_fuzzy_matches += pool.map(partial(fuzzy_skills_match_in_text, 
                                                  skills_to_match= skills_to_match), 
                                     [df_nos_local.iloc[ix] for ix in 
                                              range(istart,min(istart+100, len(df_nos_local)))])
    print('Total time elapsed (in seconds): ', time.time()-t0)
    return all_fuzzy_matches

#%
#if EXECUTE:
#    all_fuzzy_matches_nesta = extract_fuzzy_matches(df_nos2, 'nesta')
#    all_fuzzy_matches_emsi = extract_fuzzy_matches(df_nos2, 'emsi')

#%%   
def exact_skills_from_standards(df_nos_local, which_tax = 'external'):
    all_exact_matches = extract_exact_matches(df_nos_local, which_tax)  
    return all_exact_matches

def fuzzy_skills_from_standards(df_nos_local, which_tax = 'external'):
    all_fuzzy_matches = extract_fuzzy_matches(df_nos_local, which_tax)
    return all_fuzzy_matches

