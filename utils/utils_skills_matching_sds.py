#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:52:47 2019

@author: stefgarasto
"""

import pickle
import pandas as pd
import time
from fuzzywuzzy import process, fuzz
import multiprocessing
from functools import partial
from contextlib import contextmanager
#from utils_skills_clusters import bottom_layer, skills_ext, df_match_final
#from utils_skills_clusters import skills_ext_long, nesta_skills, load_and_process_clusters
#from map_NOS_to_pathways_utils import *

from collections import OrderedDict, Counter
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import torch
#import text_cleaning_util
from tqdm import tqdm

#%% general functions
@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

#%% ------------------ EMBEDDINGS ------------------ %

#%%
# Load a pre-trained BERT model and set it to "evaluation" mode
model_bert = BertModel.from_pretrained('bert-base-uncased')
model_bert.eval()

# Load pre-trained tokenizer, i.e., vocabulary;
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

def generate_embeddings(taxonomy_skills):
    '''
    Returns a sentence embedding using BERT for each skill in the list.

    Args:
        taxonomy_skills: list of skills
    '''
    if clus_name is not None:
        assert(bottom_layer is not None)
    bert_size = bert_embedding('test sentence')[0].size
    tax_embeddings = np.zeros((len(taxonomy_skills),bert_size))

    print("Generating embeddings for {} skills".format(len(taxonomy_skills)))

    t0 = time.time()
    for index, descr in tqdm(enumerate(taxonomy_skills),
                    total = len(taxonomy_skills)):
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

        tax_embeddings[index,] = skill_embedding

        if index%200 == 1:
            print('Got to skill number {} in {:.4f}'.format(index,(time.time()-t0)/60))
    print(f'All done {(time.time()-t0)/60:.4f}')

    return tax_embeddings

def generate_cluster_embeddings(taxonomy_skills, tax_embeddings,
        clus_names, bottom_layer):
    #%% average nesta embeddings according to which cluster they belong to
    '''
    Returns an average embedding per cluster based on the embeddings of the
    skills belonging to that cluster

    Args:
        taxonomy_skills: list of skills
        tax_embeddings: array of skill embeddings
        clus_names = list of cluster names
        bottom_layer = dictionary containing the cluster name
        for each taxonomy skill (skills are the keys of the dictionary)
    '''

    # get size of the BERT vectors
    bert_size = bert_embedding('test sentence')[0].size
    comparison_vectors_bert = OrderedDict()
    for clus_name in clus_names:
        comparison_vectors_bert[clus_name] = {'values': np.zeros((bert_size)), 'N': 0}

    # add each skill vector to the proper cluster
    # (yes, it'd be better if I worked with a custom dataframe)
    for i,tax_skill in enumerate(taxonomy_skills):
        # need to change this line, maybe?
        clus_name = bottom_layer[taxonomy_skill]
        if clus_name=='condition aneurysm':
            clus_name = 'treatment of aneurysms'
        comparison_vectors_bert[clus_name]['values'] += nesta_embeddings[i]
        comparison_vectors_bert[clus_name]['N']+=1

    for clus_name in clus_names:
        if clus_name=='condition aneurysm':
            clus_name = 'treatment of aneurysms'
        comparison_vectors_bert[clus_name]['values'] /= comparison_vectors_bert[clus_name]['N']

    # join everything into an array and return
    return np.stack([comparison_vectors_bert[k]['values'] for k in
                                         comparison_vectors_bert.keys()])
#%%
'''# Generate sentence-level embeddings for each taxonomy skill + cluster averages
tax_embeddings = generate_embeddings(taxonomy_skills)
comparison_vectors_bert = generate_cluster_embeddings(taxonomy_skills,tax_embeddings,
    clus_names = clus_names, bottom_layer = bottom_layer)

#%%
# Generate sentence-level embeddings for each external skill

ext_embeddings = np.zeros((len(skills_ext_left),768))
embeddings_to_match = generate_embeddings(skills_to_match)'''


#% -------------- MATCHING FUNCTIONS -------------------
#TODO: test all these functions
#%%
'''
Strategy 1
# look for all fuzzy matches between skills - if there are good ones, I already know
# the cluster
'''
#%
def check_fuzzy_matches(skills_to_match,taxonomy_skills):

    def execute_fuzzy_matches(skill_to_match):
        full_results = process.extractBests(skill_to_match, taxonomy_skills, score_cutoff = 50,
                               scorer = fuzz.ratio, limit = 100)
        good_results = [t for t in full_results if t[1]>=95]
        return good_results, full_results

    # check for partial matches: for this skills I know which cluster they belong to
    t0 = time.time()
    Nchecks = len(skills_to_match)*len(taxonomy_skills)
    with poolcontext(processes=4) as pool:
        fuzzy_matches_skill = []
        full_fuzzy_matches_skills = []
        for istart in range(0,len(skills_to_match),100):
            print(istart, time.time()-t0)
            tmp = pool.map(execute_fuzzy_matches,
                    [skills_to_match[i] for i in range(istart,min(istart+100, len(skills_to_match)))])
            fuzzy_matches_skills += [t[0] for t in tmp]
            full_fuzzy_matches_skills += [t[1] for t in tmp]
    print(f'Time elapsed to check {Nchecks} partial matches: {(time.time()-t0)/60:.3f} minutes')

    #%
    good_fuzzy_matches = {}
    for ix,t in enumerate(fuzzy_matches_skills):
        if len(t):
            good_fuzzy_matches[skills_to_match[ix]] = [t[0][0]] #[tt[0] for tt in t]
    return pd.DataFrame.from_dict(good_fuzzy_matches,
                                                orient = 'index', columns = ['fuzzy_skills'])

#%%
'''
Strategy 2.
Look for low WMD distance between skills.
Note: not sure it's worth it

'''

def check_wmd_matches(skills_to_match,taxonomy_skills):
    def execute_wmd_matches(skills_tuple):
        out = [model.wmdistance(skills_tuple[0], comparison_skill) for
               comparison_skill in skills_tuple[1]]
        return out

    t0 = time.time()
    Nchecks = 0
    with poolcontext(processes=4) as pool:
        wmd_matches_skills = []
        for istart in range(0,len(skills_to_match),100):
            print(istart, time.time()-t0)
            if isinstance(taxonomy_skills,dict):
                # I might have a different subset of taxonomy skills to check
                # for each skill to match
                compute_list = [(skills_to_match[i], taxonomy_skills[skills_to_match[i]]) for i in
                                range(istart,min(istart+100, len(skills_to_match)))]
                    #[t[0] for t in full_fuzzy_matches_nesta_ext[i]]) for i in
                    #                      range(istart,min(istart+100, len(skills_to_match)))]
            else:
                compute_list = [(skills_to_match[i], taxonomy_skills) for i in
                        range(istart,min(istart+100, len(skills_to_match)))]
            Nchecks += sum([len(t[1]) for t in compute_list])
            wmd_matches_skills += pool.map(execute_wmd_matches, compute_list)
    print(f'Time elapsed to check {Nchecks} partial matches: {(time.time()-t0)/60:.3f} minutes')

    #% extract good matches
    good_wmd_matches = {}
    for ix,t in enumerate(wmd_matches_skills):
        # wmd_matches_skills should be the same length as skills_to_match
        if (len(t)==0): # or (skills_ext[ix] in good_fuzzy_matches.index):
            continue
        wmd_argmin = np.argmin(t)
        if t[wmd_argmin]<.6:
            if isinstance(taxonomy_skills,dict):
                tmp = taxonomy_skills[skills_to_match[ix]][wmd_argmin] #full_fuzzy_matches_nesta_ext[ix][wmd_argmin]
            else:
                tmp = taxonomy_skills[wmd_argmin]
            try:
                good_wmd_matches[skills_ext[ix]] = [tmp[0]]
            except:
                good_wmd_matches[skills_ext[ix]] = [tmp]
    return pd.DataFrame.from_dict(good_wmd_matches,
                                             orient = 'index', columns = ['wmd_skills'])
'''
Functions starting with d_ are likely deprecated
'''

#%
'''
Strategy 3.
Look for best cosine similarity with all bottom layer skills using Word2Vec
-- DEPRECATED --

'''

def d_compute_embeddings(taxonomy_skills,model):
    tax_skills_embeddings = np.vstack([sentence_to_vectors_nofile(taxonomy_skill,model)[0] for
                           taxonomy_skill in taxonomy_skills])
    return tax_skills_embeddings

def d_execute_cosine_matches(skills_tuple):
    skill_to_match, tax_skills_embeddings, model = skills_tuple
    out = highest_similarity_threshold(sentence_to_vectors_nofile(skill_to_match,model)[0],
                                tax_skills_embeddings, taxonomy_skills,th=0.9)
    return out

def d_check_skill_cosine_matches(skills_to_match,tax_skills_embeddings,model):
    t0 = time.time()
    with poolcontext(processes=4) as pool:
        cosine_matches_skills = []
        for istart in range(0,len(skills_to_match),100):
            print(istart, time.time()-t0)
            compute_list = [skills_to_match[i] for i in range(istart,min(istart+100,
                            len(skills_to_match)))]
            cosine_matches_skills += pool.map(execute_cosine_matches, compute_list)
    print(f'Time spent matching all skills via cosine similarity: {(time.time()-t0)/60:.4f} minutes')

    #% extract good matches
    good_cosine_matches = {}
    for ix,t in enumerate(cosine_matches_skills):
        if len(t)>0:
            good_cosine_matches[skills_to_match[ix]] = t[0]
    return pd.DataFrame.from_dict(good_cosine_matches,
                                        orient = 'index', columns = ['cosine_skills'])

    #print(len(good_cosine_matches))


#%%
'''
Strategy 4.
Look for best cosine similarity directly with the skills cluster

'''

#%%
def d_check_cluster_matches(skill_to_match):
    out = highest_similarity_threshold(sentence_to_vectors_nofile(skill_to_match,model)[0],
                                comparison_vecs, clus_names,th=0.7)
    return out

def d_check_cluster_cosine_matches():
    # INCORRECT FUNCTION
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
# keep the one with full consensus among the four methods using word embeddings
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
In total, it gives around 14k skills (from 24k in the beginning)
'''
#%%
# save the final dataframe
matches_dir = '/Users/stefgarasto/Google Drive/Documents/results/NOS/nos_vs_skills/nos_vs_taxonomy'
with open(matches_dir + '/final_matches_nesta_vs_ext.pickle','wb') as f:
    pickle.dump(skills_matches_consensus,f)


## Also, in case I need to load previous results:
#with open(matches_dir + '/cosine_bert_matches_nesta_vs_ext_short.pickle','wb') as f:
#    pickle.dump(good_cosine_bert_matches,f)
