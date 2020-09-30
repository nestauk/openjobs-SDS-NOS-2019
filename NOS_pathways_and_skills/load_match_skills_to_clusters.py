#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 10:05:44 2019

@author: stefgarasto

Script to load the match between skills and skill clusters 
and to perform some light cleaning
"""

import pickle
from copy import deepcopy

select_punct_skills = set('!"#$%&\()*+,./:;<=>?@[\\]^_`{|}~') #here "'","-" are not included
extra_chars = set('•’”“µ¾âãéˆﬁ[€™¢±ï…˜')
all_select_chars_skills = select_punct_skills.union(extra_chars).union('\'')


def replace_punctuation_skills(s):
    """
    Takes string as input.
    Removes punctuation from a string if the character is in select_punct.
    Returns a string.
   >>> replace_punctuation('sales executives/ - london')
   'sales executives   london'
    """
    for i in set(all_select_chars_skills): #set(select_punct):
        if i in s:
            s = s.replace(i, ' ')
    return s

def tidy_desc_with_pos_skills(desc,pof):
    '''
    
    '''
    clean_data = desc.replace('\r\n', '').replace('\xa0', '').lower()
    nopunct = replace_punctuation_skills(clean_data)
    # add part of speech tagging
    nopunct = nltk.pos_tag(nopunct.split())
    nopunct = [t for t in nopunct if t[1] in pos_to_wornet_dict.keys()]
    lemm = lemmatise_pruned(nopunct, pof)
    return lemm #' '.join(lemm)

#%%
# load the skills from the directory "location_dir"
location_dir = '/Users/stefgarasto/Google Drive/Documents/results/NOS/nos_vs_skills/nos_vs_taxonomy'
with open(location_dir + '/final_matches_nesta_vs_ext.pickle','rb') as f:
    skills_matches_consensus_ext = pickle.load(f)

with open(location_dir + '/final_matches_nesta_vs_ext_emsi.pickle','rb') as f:
    skills_matches_consensus_emsi = pickle.load(f)

skills_matches_consensus = skills_matches_consensus_ext.append(
        skills_matches_consensus_emsi)

# remove duplicated skills
skills_matches_consensus = skills_matches_consensus.reset_index().drop_duplicates().set_index('index')

# create the lemmatised version
skills_matches_consensus['lemmatised'] = skills_matches_consensus.index.map(lambda x: 
    ' '.join(tidy_desc_with_pos_skills(x,'all')))
skills_matches_consensus['lemmatised'] = skills_matches_consensus['lemmatised'].map(
        lambda x : x.replace('\'',''))

# remove duplicates based on the lemmatised version - this will ignore the index and
# check whether there is any row with same cluster and lemmatise version
skills_matches_consensus2 = skills_matches_consensus2.drop_duplicates()

#%% put lemmatised skills in their own dataframe
#new_skills_lemmas = skills_matches_consensus2['lemmatised'].map
skills_lemmas = skills_matches_consensus2.reset_index().rename(
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