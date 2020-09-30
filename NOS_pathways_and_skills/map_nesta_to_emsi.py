#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 19:36:44 2019

@author: stefgarasto
"""
from utils_skills_clusters import emsi_skills, bottom_layer
from fuzzywuzzy import process, fuzz
import time

#%%
nesta_skills = list(bottom_layer.keys())
#%%

nesta_to_emsi_partial = {}
emsi_skills_names = [t['name'].lower() for t in emsi_skills if t['name']!='IT++']

#%%
t0 = time.time()
for ix in range(7000,10218):
    out = process.extract(nesta_skills[ix],emsi_skills_names,limit= 1,scorer=fuzz.partial_ratio)
    nesta_to_emsi_partial[nesta_skills[ix]] = out
    if out[0][1]>95:
        print(nesta_skills[ix])
        print(out[0][0])
print(time.time()-t0)

#%%
nesta_to_emsi_partial2={}
for ix in range(len(nesta_skills)):
    tmp = nesta_to_emsi_partial[nesta_skills[ix]]
    nesta_to_emsi_partial2[nesta_skills[ix]] = {'best_match': tmp[0][0], 
                         'best_match_value': tmp[0][1]}
nesta_to_emsi_partial2 = pd.DataFrame.from_dict(nesta_to_emsi_partial2, orient='index')
nesta_to_emsi_partial2.to_csv(
        ''.join(['/Users/stefgarasto/Google Drive/Documents/results/NOS/',
                 'nos_vs_skills/nesta_vs_emsi/nesta_to_emsi_partial.csv']))

#%%
nesta_to_emsi_full = {}

#%%
t0 = time.time()
for ix in range(1000,10218):
    if nesta_to_emsi_partial[nesta_skills[ix]][0][1]>95:
        out = process.extract(nesta_skills[ix],emsi_skills_names,limit= 1)
        nesta_to_emsi_full[nesta_skills[ix]] = out
    if ix%500 == 1:
        print('Got to index ',ix)
print(time.time()-t0)

#%%
nesta_to_emsi_full2={}
for ix in list(nesta_to_emsi_full.keys()):
    tmp = nesta_to_emsi_full[ix]
    nesta_to_emsi_full2[ix] = {'best_match': tmp[0][0], 
                         'best_match_value': tmp[0][1]}
nesta_to_emsi_full2 = pd.DataFrame.from_dict(nesta_to_emsi_full2, orient='index')
nesta_to_emsi_full2.to_csv(
        ''.join(['/Users/stefgarasto/Google Drive/Documents/results/NOS/',
                 'nos_vs_skills/nesta_vs_emsi/nesta_to_emsi_full.csv']))
   

#%%
'''
# check whether those matched to 90 have a better match
no_better_matches = {}
yes_better_matches = {}
match90 = nesta_to_emsi_full2[nesta_to_emsi_full2['best_match_value']==90]
for name,row in match90.iterrows():
    tmp = nesta_to_emsi_full2[nesta_to_emsi_full2['best_match']==row['best_match']]
    if tmp['best_match_value'].max()<=90:
        no_better_matches[name] = {'best_match': row['best_match'],
                         'best_match_value': row['best_match_value']}
    else:
        best = tmp['best_match_value'].idxmax()
        yes_better_matches[name] = {'best_match': tmp.loc[best]['best_match'],
                         'best_match_value': tmp.loc[best]['best_match_value']}
'''
    
#%%
# only select those with >=90 match
nesta_to_emsi_full3 = nesta_to_emsi_full2[nesta_to_emsi_full2['best_match_value']>89]
import networkx as nx
from networkx.algorithms import bipartite
nesta_translator = {}
emsi_translator = {}
G = nx.Graph()
emsi_matched= list(nesta_to_emsi_full3['best_match'].value_counts().index)

#for ix,name in enumerate(nesta_to_emsi_full2.index):
#    nesta_translator['nesta_{}'.format(ix)] = name
#for ix,name in enumerate(emsi_matched):
#    emsi_translator['emsi_{}'.format(ix)] = name   
#G.add_nodes_from(['nesta_{}'.format(ix) for ix in range(len(nesta_to_emsi_full))], bipartite = 0)
#G.add_nodes_from(['emsi_{}'.format(ix) for ix in range(len(emsi_matched))], bipartite = 1)
#for ix in range(len(nesta_to_emsi_full)):
#    emsi_ix = emsi_matched.index(nesta_to_emsi_full2.iloc[ix]['best_match'])
#    G.add_edges_from([('nesta_{}'.format(ix), 'emsi_{}'.format(emsi_ix))],
#                      weight = nesta_to_emsi_full2.iloc[ix]['best_match_value'])

G.add_nodes_from(['nesta_' + t for t in list(nesta_to_emsi_full3.index)], bipartite = 0)
G.add_nodes_from(['emsi_'+t for t in emsi_matched], bipartite = 1)
for ix in nesta_to_emsi_full3.index:
    G.add_edges_from([('nesta_'+ix, 'emsi_'+nesta_to_emsi_full3.loc[ix]['best_match'])],
                      weight = nesta_to_emsi_full3.loc[ix]['best_match_value'])

B = list(nx.maximal_matching(G))
#C = [(nesta_translator[t[0]],emsi_translator[t[1]]) for t in B]
C = [(t[0].split('_')[1],t[1].split('_')[1],
      fuzz.WRatio(t[0].split('_')[1],t[1].split('_')[1])) for t in B]
df_match = pd.DataFrame(C, columns = ['nesta','emsi','match'])
df_match = df_match.sort_values('match')
df_match.to_csv(''.join(['/Users/stefgarasto/Google Drive/Documents/results/NOS/',
                 'nos_vs_skills/nesta_vs_emsi/nesta_to_emsi_bipartite_match.csv']))
    
