#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:51:41 2019

@author: stefgarasto
"""

import pickle
import pandas as pd
import time
from fuzzywuzzy import process, fuzz
import multiprocessing
from functools import partial
from contextlib import contextmanager
from utils_skills_clusters import bottom_layer, df_match_final, emsi_skills
from utils_skills_clusters import nesta_skills, load_and_process_clusters
from map_NOS_to_pathways_utils import *

from collections import OrderedDict, Counter
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import torch
#import text_cleaning_util
from tqdm.notebook import tqdm

'''
Repeat other analysis for EMSI skills
'''
skills_ext= [t['name'].lower() for t in emsi_skills]

#%%
'''
Strategy 1
# look for all fuzzy matches - if there are good ones, I already know 
# the cluster
'''
#%
@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def check_fuzzy_matches(i):
    full_results = process.extractBests(skills_ext[i], nesta_skills, score_cutoff = 50, 
                           scorer = fuzz.ratio, limit = 100)
    good_results = [t for t in full_results if t[1]>=95]
    return good_results, full_results

# check for partial matches: for this skills I know which cluster they belong to
t0 = time.time()
with poolcontext(processes=4) as pool:
    fuzzy_matches_nesta_ext = []
    full_fuzzy_matches_nesta_ext = []
    for istart in range(0,len(skills_ext),100):
        print(istart, time.time()-t0)
        tmp = pool.map(check_fuzzy_matches, 
                                 range(istart,min(istart+100, len(skills_ext))))
        fuzzy_matches_nesta_ext += [t[0] for t in tmp]
        full_fuzzy_matches_nesta_ext += [t[1] for t in tmp]
        
print(time.time()-t0)

#%
good_fuzzy_matches = {}
for ix,t in enumerate(fuzzy_matches_nesta_ext):
    if len(t):
        good_fuzzy_matches[skills_ext[ix]] = [t[0][0]] #[tt[0] for tt in t]
good_fuzzy_matches = pd.DataFrame.from_dict(good_fuzzy_matches, 
                                            orient = 'index', columns = ['fuzzy_skills'])

#%%
'''
Strategy 2.
Look for low WMD distance between skills.
It's not worth it - I'll just add manually some stuff it identified 

'''
manual_additions = {
  'ar system administration': 'system administration',
  'base station subsystem': 'base station system bss',
  'cats applicant tracking system': 'applicant tracking system',
  'capability maturity model integration': 'capability maturity model integration cmmi',
  'cauterization': 'cauterisation',
  'content management systems': 'content management systems cms',
  'customer relationship management': 'customer relationship management crm',
  'data visualization': 'data visualisation',
  'design tool': 'tool design',
  'direct electronic data interchange': 'electronic data interchange',
  'distributed component object model': 'distributed component object model dcom',
  'drilling engineering': 'drill engineering',
  'electron optics': 'optoelectronics',
  'electronic dental records': 'electronic health records',
  'electronic news production systems': 'electronic news production system enps',
  'enterprise master patient index': 'enterprise master patient index empi',
  'enterprise resource planning': 'enterprise resource planning erp',
  'esophagogastroduodenoscopy': 'oesophagogastroduodenoscopy egd',
  'failure mode and effects analysis': 'failure modes and effects analysis fmea',
  'generally accepted auditing standards': 'generally accepted auditing standards gaas',
  'hp 3000': 'hp3000',
  'hp 9000': 'hp9000',
  'hp quicktest professional': 'quick test professional qtp',
  'health information management': 'health information management him',
  'health information technology': 'health information technology hit',
  'high throughput screening': 'high throughput screening hts',
  'ibm resource access control facility': 'ibm resource access control facility racf',
  'immunoglobulin g': 'immunoglobulin',
  'integrated development environments': 'integrated development environment ide',
  'international traffic in arms regulations': 'international traffic in arms regulations itar',
  'ionizing radiation': 'ionising radiation',
  'java naming and directory interface': 'java naming and directory interface jndi',
  'joint mobilization': 'joint mobilisation',
  'lyophilization': 'lyophilisation',
  'management': 'team management',
  'manufacturing resource planning': 'manufacturing resource planning mrp',
  'mass customization': 'mass customisation',
  'medicare part d': 'medicare part b',
  'micro-opto-electro-mechanical systems': 'micro-opto-electro-mechanical systems moems',
  'monitoring results': 'results monitoring',
  'neonatal intensive care unit': 'neonatal intensive care unit nicu',
  'occupational safety and health': 'occupational health and safety',
  'online analytical processing': 'online analytical processing olap',
  'oracle business intelligence enterprise edition': 'oracle business intelligence enterprise edition obiee',
  'original equipment manufacturer': 'original equipment manufacturer oem',
  'peoplesoft people tools': 'peopletools',
  'positron emission tomography': 'positron emission tomography pet',
  'process manufacturing': 'manufacturing processes',
  'project portfolio management': 'product portfolio management',
  'rna sequencing': 'dna sequencing',
  'real estate settlement procedures act': 'real estate settlement procedures act respa',
  'regulation of gene expression': 'gene expression regulation',
  'sap production planning': 'sap pp production planning',
  'smtp (simple mail transfer protocol)': 'simple mail transfer protocol smtp',
  'setting appointments': 'appointment setting',
  'team performance management': 'performance management',
   'time management': 'team management',
  'transportation systems management': 'transportation management systems',
  "traveler's checks": 'travellers checks',
  'tree view': 'treeview',
  'turbotax': 'turbo tax',
  'unified medical language systems': 'unified medical language system umls',
  'visual basic for applications': 'visual basic for applications vba',
  'wildlife observation': 'wildlife conservation',
}

def check_wmd_matches(skills_tuple):
    out = [model.wmdistance(skills_tuple[0], comparison_skill) for 
           comparison_skill in skills_tuple[1][:50]]
    return out

'''
t0 = time.time()
with poolcontext(processes=4) as pool:
    wmd_matches_nesta_ext = []
    for istart in range(0,len(skills_ext),100):
        print(istart, time.time()-t0)
        compute_list = [(skills_ext[i],[t[0] for t in full_fuzzy_matches_nesta_ext[i]]) for i in 
                                      range(istart,min(istart+100, len(skills_ext)))]
        wmd_matches_nesta_ext += pool.map(check_wmd_matches, compute_list)
print(time.time()-t0)

#% extract good matches
good_wmd_matches = {}
for ix,t in enumerate(wmd_matches_nesta_ext):
    if (len(t)==0) or (skills_ext[ix] in good_fuzzy_matches.index):
        continue
    wmd_argmin = np.argmin(t)
    if t[wmd_argmin]<.6:
        tmp = full_fuzzy_matches_nesta_ext[ix][wmd_argmin]
        good_wmd_matches[skills_ext[ix]] = [tmp[0]]
good_wmd_matches = pd.DataFrame.from_dict(good_wmd_matches, 
                                         orient = 'index', columns = ['wmd_skills'])
'''
good_wmd_matches = pd.DataFrame.from_dict(manual_additions, orient = 'index', 
                                          columns = ['fuzzy_skills'])

#%%
# when you're ready to go to the next step
skills_ext_left= [t for t in skills_ext if t not in list(good_fuzzy_matches.index)]
skills_ext_left= [t for t in skills_ext_left if t not in list(good_wmd_matches.index)]

#%
'''
Strategy 3.
Look for best cosine similarity with all bottom layer skills

'''
nesta_skills_embeddings = np.vstack([sentence_to_vectors_nofile(nesta_skill,model)[0] for 
                           nesta_skill in nesta_skills])

def check_cosine_matches(skill_to_match):
    out = highest_similarity_threshold(sentence_to_vectors_nofile(skill_to_match,model)[0],
                                nesta_skills_embeddings, nesta_skills,th=0.9)
    return out

t0 = time.time()
with poolcontext(processes=4) as pool:
    cosine_matches_nesta_ext = []
    for istart in range(0,len(skills_ext_left),100):
        print(istart, time.time()-t0)
        compute_list = [skills_ext_left[i] for i in range(istart,min(istart+100, 
                        len(skills_ext_left)))]
        cosine_matches_nesta_ext += pool.map(check_cosine_matches, compute_list)
print(time.time()-t0)

#% extract good matches
good_cosine_matches = {}
for ix,t in enumerate(cosine_matches_nesta_ext):
    if len(t)>0:
        good_cosine_matches[skills_ext_left[ix]] = t[0]
good_cosine_matches = pd.DataFrame.from_dict(good_cosine_matches, 
                                    orient = 'index', columns = ['cosine_skills'])

print(len(good_cosine_matches))


#%%
'''
Strategy 4.
Look for best cosine similarity directly with the skills cluster

'''
clus_names, comparison_vecs, tmp = load_and_process_clusters(model, ENG = False)
skill_cluster_vecs, full_skill_cluster_vecs = tmp

#%%
def check_cluster_matches(skill_to_match):
    out = highest_similarity_threshold(sentence_to_vectors_nofile(skill_to_match,model)[0],
                                comparison_vecs, clus_names,th=0.7)
    return out

t0 = time.time()
with poolcontext(processes=4) as pool:
    cluster_matches_nesta_ext = []
    for istart in range(0,len(skills_ext_left),100):
        print(istart, time.time()-t0)
        compute_list = [skills_ext_left[i] for i in range(istart,min(istart+100, 
                        len(skills_ext_left)))]
        cluster_matches_nesta_ext += pool.map(check_cluster_matches, compute_list)
print(time.time()-t0)

#% extract good matches
good_cluster_matches = {}
for ix,t in enumerate(cluster_matches_nesta_ext):
    if len(t)>0:
        good_cluster_matches[skills_ext_left[ix]] = t[0]
good_cluster_matches = pd.DataFrame.from_dict(good_cluster_matches, 
                                        orient = 'index', columns=['cluster'])

print(len(good_cluster_matches))



#%%
# Load a pre-trained BERT model and set it to "evaluation" mode
# note that, when this is ran for the first time, a download of the model will be initiated (approx. 400MB)
model_bert = BertModel.from_pretrained('bert-base-uncased')
model_bert.eval()

# Load pre-trained tokenizer, i.e., vocabulary;
# this will also initiate a small download when evaluated for the first time
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def bert_embedding(sent):

    """
    Returns a sentence embedding using BERT. 
    
    Args:
        sent: string containing a single sentence
    """
    
    # Add BERT-specific markers for beginning and end of the sentence
    marked_sent = "[CLS] " + sent + " [SEP]"
        
    # Tokenize sentence in a way that BERT model understands 
    tokenized_text = tokenizer.tokenize(marked_sent)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        
    # Mark which tokens belong to which sentence (in our case, we only have single sentences)
    segments_ids = [1] * len(tokenized_text)
        
    # Convert data into pytorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensor = torch.tensor([segments_ids])
        
    # Obtain the token representations with the dimensions [layer=0...11][batch=0][token][features=768]
    with torch.no_grad():
        encoded_layers, _ = model_bert(tokens_tensor, segments_tensor)
        
    # Create the sentence embedding 
    sent_embedding = torch.mean(encoded_layers[11], 1)
    sent_embedding = sent_embedding[0].numpy()
    
    return sent_embedding, tokenized_text, encoded_layers[11].numpy()[0]

#%%
# Generate sentence-level embeddings for each nesta skill and cluster

nesta_embeddings = np.zeros((len(nesta_skills),768))

print("Generating embeddings for {} skills".format(len(nesta_skills)))

t0 = time.time()
for index, descr in enumerate(nesta_skills):
    #tqdm(enumerate(nesta_skills), total=len(nesta_skills)):
    
    #descr = row['description']
    
    # Tokenize description into a list of separate sentences
    sentences = nltk.tokenize.sent_tokenize(descr)
    
    # Generate embedding for each sentence in skill's description
    sentence_embeddings = []    
    for sent in sentences:  
        sent_embedding, _, _ = bert_embedding(sent)        
        sentence_embeddings.append(sent_embedding)
    
    # If a skill is described by more than one sentence, average across all sentence embeddings
    skill_embedding = np.mean(sentence_embeddings,0)
    
    nesta_embeddings[index,] = skill_embedding
    
    if index%200 == 1:
        print('Got to skill number {} in {}'.format(index,time.time()-t0))
print('All done',time.time()-t0)

#%% average nesta embeddings according to which cluster they belong to
comparison_vectors_bert = OrderedDict()
for clus_name in clus_names:
    comparison_vectors_bert[clus_name] = {'values': np.zeros((768)), 'N': 0}
    
for i,nesta_skill in enumerate(nesta_skills):
    clus_name = bottom_layer[nesta_skill]
    if clus_name=='condition aneurysm':
        clus_name = 'treatment of aneurysms'
    comparison_vectors_bert[clus_name]['values'] += nesta_embeddings[i]
    comparison_vectors_bert[clus_name]['N']+=1

for clus_name in clus_names:
    if clus_name=='condition aneurysm':
        clus_name = 'treatment of aneurysms'
    comparison_vectors_bert[clus_name]['values'] /= comparison_vectors_bert[clus_name]['N']

# join everything into an array
comparison_vectors_bert = np.stack([comparison_vectors_bert[k]['values'] for k in 
                                     comparison_vectors_bert.keys()])
            
#%%
# Generate sentence-level embeddings for each external skill

ext_embeddings = np.zeros((len(skills_ext_left),768))

print("Generating embeddings for {} external skills".format(len(skills_ext_left)))

t0 = time.time()
for index, descr in enumerate(skills_ext_left): 
    #tqdm(enumerate(skills_ext_left), total=len(skills_ext_left)):
    
    #descr = row['description']
    
    # Tokenize description into a list of separate sentences
    sentences = nltk.tokenize.sent_tokenize(descr)
    
    # Generate embedding for each sentence in skill's description
    sentence_embeddings = []    
    for sent in sentences:  
        sent_embedding, _, _ = bert_embedding(sent)        
        sentence_embeddings.append(sent_embedding)
    
    # If a skill is described by more than one sentence, average across all sentence embeddings
    skill_embedding = np.mean(sentence_embeddings,0)
    
    ext_embeddings[index,] = skill_embedding
    
    if index%200 == 1:
        print('Got to skill number {} in {}'.format(index,time.time()-t0))
print('All done',time.time()-t0)

#%%
'''
Strategy 5.
Look for best cosine similarity with all bottom layer skills using BERT

'''

def check_cosine_bert_matches(embedding_to_match):
    out = highest_similarity_threshold(embedding_to_match,
                                nesta_embeddings, nesta_skills, th=0.85)
    return out

t0 = time.time()
with poolcontext(processes=4) as pool:
    cosine_bert_matches_nesta_ext = []
    for istart in range(0,len(ext_embeddings),100):
        print(istart, time.time()-t0)
        compute_list = [ext_embeddings[i] for i in range(istart,min(istart+100, 
                        len(ext_embeddings)))]
        cosine_bert_matches_nesta_ext += pool.map(check_cosine_bert_matches, compute_list)
print(time.time()-t0)

#% extract good matches
good_cosine_bert_matches = {}
for ix,t in enumerate(cosine_bert_matches_nesta_ext):
    if len(t)>0:
        good_cosine_bert_matches[skills_ext_left[ix]] = t[0]
good_cosine_bert_matches = pd.DataFrame.from_dict(good_cosine_bert_matches, 
                                    orient = 'index', columns = ['cosine_bert_skills'])

print(len(good_cosine_bert_matches))

#%%
'''
Strategy 6.
Look for best cosine similarity directly with the skills cluster using BERT

'''

def check_cluster_bert_matches(embedding_to_match):
    out = highest_similarity_threshold(embedding_to_match,
                                comparison_vectors_bert, clus_names,th=0.7)
    return out

t0 = time.time()
with poolcontext(processes=4) as pool:
    cluster_bert_matches_nesta_ext = []
    for istart in range(0,len(ext_embeddings),100):
        print(istart, time.time()-t0)
        compute_list = [ext_embeddings[i] for i in range(istart,min(istart+100, 
                        len(ext_embeddings)))]
        cluster_bert_matches_nesta_ext += pool.map(check_cluster_bert_matches, compute_list)
print(time.time()-t0)

#% extract good matches
good_cluster_bert_matches = {}
for ix,t in enumerate(cluster_bert_matches_nesta_ext):
    if len(t)>0:
        good_cluster_bert_matches[skills_ext_left[ix]] = t[0]
good_cluster_bert_matches = pd.DataFrame.from_dict(good_cluster_bert_matches, 
                                    orient = 'index', columns = ['cluster_bert'])

print(len(good_cluster_bert_matches))

#%%
'''
Join all the results together

'''
def skill_to_cluster_nan(x):
    if isinstance(x,float): 
        if np.isnan(x):
            return x
    else:
        tmp= bottom_layer[x]
        if tmp == 'condition aneurysm':
            tmp = 'treatment of aneurysms'
        return tmp

# the one with high match between skills are certain
skills_matches_certain = good_wmd_matches.append(good_fuzzy_matches)
skills_matches_certain['consensus_cluster'] = skills_matches_certain['fuzzy_skills'].map(
        skill_to_cluster_nan)
# these other need some consensus check
skills_matches_df = good_cosine_matches.join(good_cosine_bert_matches, how ='outer'
                            ).join(good_cluster_matches, how ='outer'
                            ).join(good_cluster_bert_matches, how ='outer')

# transform the links to Nesta skills into Nesta cluster

#skills_matches_df['cluster_fuzzy'] = skills_matches_df['fuzzy_skills'].map(
#        skill_to_cluster_nan)
#skills_matches_df['cluster_wmd'] = skills_matches_df['wmd_skills'].map(
#        skill_to_cluster_nan)
skills_matches_df['cluster_cosine'] = skills_matches_df['cosine_skills'].map(
        skill_to_cluster_nan)
skills_matches_df['cluster_cosine_bert'] = skills_matches_df['cosine_bert_skills'].map(
        skill_to_cluster_nan)
# retain the clusters columns only
skills_matches_df = skills_matches_df[[t for t in skills_matches_df.columns if 'cluster' in t]]

#%%
# keep the one with full consensus among the four methods
def keep_full_consensus(row):
    clus_counter = Counter(row[row.notna()]).most_common()
    if len(clus_counter)==1:
        return clus_counter[0][0]
    else:
        return []
    
skills_matches_df['consensus_cluster'] = skills_matches_df.apply(keep_full_consensus, axis = 1)

#%%
# keep the good ones and the ones the algorithms agree on
skills_matches_consensus = skills_matches_certain['consensus_cluster'].append(
        skills_matches_df['consensus_cluster'][skills_matches_df['consensus_cluster'
                         ].map(lambda x: len(x)>1)])

'''
In total, it gives around 17.5k skills (from 27k in the beginning)
'''
print('Final numbers. {} skills have been matched out of {}'.format(len(skills_matches_consensus),
      len(skills_ext)))

#%%
# save the final dataframe
matches_dir = '/Users/stefgarasto/Google Drive/Documents/results/NOS/nos_vs_skills/nos_vs_taxonomy'
with open(matches_dir + '/final_matches_nesta_vs_ext_emsi.pickle','wb') as f:
    pickle.dump(skills_matches_consensus,f)
    
    
## Also, in case I need to load previous results:
#with open(matches_dir + '/cosine_bert_matches_nesta_vs_ext_short.pickle','wb') as f:
#    pickle.dump(good_cosine_bert_matches,f)



