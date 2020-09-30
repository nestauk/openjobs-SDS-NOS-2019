#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 16:22:20 2019

@author: jdjumalieva
"""
import pandas as pd
import numpy as np
import gensim
import os
import pickle
import collections
from sklearn.feature_extraction.text import TfidfVectorizer


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

def convert_from_undersc(skill):
    '''
    convert underscores between terms in skill phrases back to spaces.
    '''
    if len(skill.split('_')) >1:
        new_i = ' '.join(skill.split('_'))
    else:
        new_i = skill
    return(new_i)

def get_mean_vec(skill_list, model):
    skill_list_conv = [convert_to_undersc(elem) for elem in skill_list]
    vector_list = [model[elem] for elem in skill_list_conv if elem in model]
    vec_array = np.asarray(vector_list)
    avg_vec = np.mean(vec_array, axis=0)
    return avg_vec

def get_average_skill_category(list_of_skills, reference_dict):
    """
    Returns top 10 categories in the averaged cosine sim array.
    """
    pruned_skills = [elem for elem in list_of_skills if elem in reference_dict]
    if len(pruned_skills):
        vec_list = [reference_dict[skill] for skill in pruned_skills]
        vec_array = np.asarray(vec_list)
        avg_vec = np.mean(vec_array, axis=0)
        sorted_vecs = np.argsort(avg_vec)[0, -10:]
        scores = [avg_vec[0,i] for i in sorted_vecs]
        categories_values = zip(sorted_vecs, scores)
        res = list(categories_values)
    else:
        res = []
    return res


def get_best_skill_category(list_of_skills, reference_dict, transversal):
    top10 = get_average_skill_category(list_of_skills, reference_dict)
    if len(top10):
        dom_specific = [elem for elem in top10 if elem[0] not in transversal]
        best = max(dom_specific, key = lambda x: x[1])
    else:
        best = (999, 0.0)
    return best
    
lookup_dir = '/Users/jdjumalieva/ESCoE/lookups/'
output_dir = '/Users/jdjumalieva/ESCoE/outputs'


df_api = pd.read_csv(os.path.join(output_dir, 'df_api.csv'),
                     encoding = 'utf-8')



model = gensim.models.Word2Vec.load(os.path.join(lookup_dir, 'w2v_model'))

vector_matrix = model.wv.syn0
list_of_terms = model.wv.index2word

lookup_terms = [convert_from_undersc(elem) for elem in list_of_terms]


def merge_cols(row):
    title = row['Title']
    desc = row['clean_standard_info2']
#    title2 = tidy_desc(title)
    res = ' '.join([title, desc])
#    res = ' '.join([title2, desc])
    return res

df_api['combined'] = df_api.apply(merge_cols, axis =1)

textfortoken= df_api['combined']
tfidf = TfidfVectorizer(ngram_range=(1,2), 
                        max_df = 0.5, 
                        min_df = 2)
tfidfm = tfidf.fit_transform(textfortoken)
feature_names = tfidf.get_feature_names()

top_terms = {}
for ix, row in df_api.iterrows():
    title = row['Title']
    top_ngrams = np.argsort(tfidfm.todense()[ix,:])
    top_features = [feature_names[elem] for elem in top_ngrams.tolist()[0][-25:]]
    top_terms[title] = top_features
    print(title, top_features)
    print('**************************************')

def prep_for_gensim(list_of_terms, some_model):
    # replace space with underscore
    new_terms = [convert_to_undersc(elem) for elem in list_of_terms]
    # check if each element in the list is in the model
    is_in = [elem for elem in new_terms if elem in some_model]
    # only return the element in the model
    return is_in

for k,v in top_terms.items():
    # check if the top terms for each document are in the gensim model
    new_top_terms = prep_for_gensim(v, model)
    # only retains the ones in the model
    top_terms[k] = new_top_terms
    print(k, new_top_terms)

#Generate lookup vecs using pre-trained GloVe model
with open(os.path.join(output_dir, 'bottom_cluster_membership.pkl'), 'rb') as infile:
    bottom_layer = pickle.load(infile)
    
skill_cluster_membership = {}
for clus in collections.Counter(bottom_layer.values()):
    cluster_skills = [elem for elem in bottom_layer if \
                      bottom_layer[elem] == clus]
    skill_cluster_membership[clus] = cluster_skills
    
skill_cluster_vecs = {}
for clus in skill_cluster_membership:
    cluster_skills = skill_cluster_membership[clus]
    new_skills = [convert_to_undersc(elem) for elem in cluster_skills]
    other_skills = [elem.split() for elem in cluster_skills if len(elem)>1]
    flat_other_skills = [item for sublist in other_skills for item in sublist]
    all_skills = new_skills + list(set(flat_other_skills))
    skills_in = [elem for elem in all_skills if elem in model]
    print(clus, len(cluster_skills), len(skills_in))
    skill_cluster_vecs[clus] = get_mean_vec(skills_in, 
                            model)

#with open(os.path.join(output_dir, 'skill_cluster_vecs_pretrained.pkl'), 'wb') as f:
#    pickle.dump(skill_cluster_vecs, f)
    
for clus in list(skill_cluster_vecs.keys())[:10]:
    print(clus)
    print(model.similar_by_vector(skill_cluster_vecs[clus]))
    print('***********')

check = [k for k,v in skill_cluster_vecs.items() if len(v.shape) == 0]


from sklearn.metrics.pairwise import cosine_similarity

comparison_vecs = np.vstack(list(skill_cluster_vecs.values()))
clus_names = list(skill_cluster_vecs.keys())


#test_skills = get_mean_vec(top_terms['Boatbuilder'],
#                                     model)

st_v_clus = {}
for k in top_terms.keys():
    test_skills = get_mean_vec(top_terms[k], model)

    sims = cosine_similarity(test_skills.reshape(1,-1), comparison_vecs)
    
    top_sims = np.argsort(sims)[:, -5:].tolist()[0]
    top_sim_vals = [sims[0, elem] for elem in top_sims]
    top_sim_clus = [clus_names[elem] for elem in top_sims]
    top_sims_res = list(zip(reversed(top_sim_clus), reversed(top_sim_vals)))
    for elem in top_sims_res:
        print(elem)
    st_v_clus[k] = top_sims_res