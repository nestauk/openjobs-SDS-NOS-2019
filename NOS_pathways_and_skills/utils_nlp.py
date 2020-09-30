#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 14:51:29 2019

@author: stefgarasto
"""

#%% imports
from utils_bg import *
import itertools
import numpy as np
import nltk
import scipy
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity


#%%
# flatten lists of lists
def flatten_lol(t):
    return list(itertools.chain.from_iterable(t))
#flatten_lol([[1,2],[3],[4,5,6]])


# In[5]:

#%%
def get_mean_vec(skill_list, model, weights= None):
    if not weights:
        weights = np.ones(len(skill_list))
    #skill_list_conv = [convert_to_undersc(elem) for elem in skill_list]
    #wvector_list = [model[elem]*weights[ix] for ix,elem in enumerate(skill_list_conv) 
    #            if elem in model]
    skill_list_conv = skill_list
    vector_list = [sentence_to_vectors_nofile(elem,model)[0]*weights[ix] for 
                    ix,elem in enumerate(skill_list_conv)]
    wvector_list = [elem*weights[ix] for ix,elem in enumerate(vector_list)]
    vec_array = np.asarray(vector_list)
    wvec_array = np.asarray(wvector_list)
    avg_vec = np.mean(wvec_array, axis=0)
    return avg_vec, vec_array


#%%
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

#%%
def get_best_skill_category(list_of_skills, reference_dict, transversal):
    top10 = get_average_skill_category(list_of_skills, reference_dict)
    if len(top10):
        dom_specific = [elem for elem in top10 if elem[0] not in transversal]
        best = max(dom_specific, key = lambda x: x[1])
    else:
        best = (999, 0.0)
    return best
    


# In[7]:
pos_to_wornet_dict = {
        'JJ': 'a',
        'JJR': 'a',
        'JJS': 'a',
        'RB': 'r',
        'RBR': 'r',
        'RBS': 'r',
        'NN': 'n',
        'NNP': 'n',
        'NNS': 'n',
        'NNPS': 'n',
        'VB': 'v',
        'VBG': 'v',
        'VBD': 'v',
        'VBN': 'v',
        'VBP': 'v',
        'VBZ': 'v'
    }

#A few functions for tyding up text
def tag_for_lemmatise(s):
    try:
        return pos_to_wornet_dict[nltk.pos_tag([s])[0][1]]
    except:
        return 'n'

#%%    
def lemmatise(title_terms):
    """
    Takes list as input.
    Removes suffixes if the new words exists in the nltk dictionary.
    The purpose of the function is to convert plural forms into singular.
    Allows some nouns to remain in plural form (the to_keep_asis is manually curated).
    Returns a list.
    >>> lemmatise(['teachers'])
    ['teacher']
    >>> lemmatise(['analytics'])
    ['analytics']
    """
    keep_asis = ['sales', 'years', 'goods', 'operations', 'systems',
                    'communications', 'events', 'loans', 'grounds',
                    'lettings', 'claims', 'accounts', 'relations',
                    'complaints', 'services']
    wnl = nltk.WordNetLemmatizer()
    processed_terms = [wnl.lemmatize(i) if i not in keep_asis else i for i in title_terms]
    #processed_terms = [wnl.lemmatize(i, pos = tag_for_lemmatise(i)) 
    #            if i not in keep_asis else i for i in title_terms]
    return processed_terms

#%%

def lemmatise_with_pos(title_terms):
    """
    Takes list as input.
    Removes suffixes if the new words exists in the nltk dictionary.
    The purpose of the function is to convert plural forms into singular.
    Allows some nouns to remain in plural form (the to_keep_asis is manually curated).
    Returns a list.
    >>> lemmatise(['teachers'])
    ['teacher']
    >>> lemmatise(['analytics'])
    ['analytics']
    """

    keep_asis = ['sales', 'years', 'goods', 'operations', 'systems',
                    'communications', 'events', 'loans', 'grounds',
                    'lettings', 'claims', 'accounts', 'relations',
                    'complaints', 'services']
    wnl = nltk.WordNetLemmatizer()
    processed_terms = [wnl.lemmatize(i, pos_to_wornet_dict[p]) if i not in keep_asis 
                       else i for i,p in title_terms]
    #processed_terms = [wnl.lemmatize(i, pos = tag_for_lemmatise(i)) 
    #            if i not in keep_asis else i for i in title_terms]
    return processed_terms

#%%
def stem_features(s, ps):
    return ps.stem(s)

#%%    
def remove_digits(s):
    """
    Takes a string as input.
    Removes digits in a string.
    Returns a string.
    >>> remove_digits('2 recruitment consultants')
    ' recruitment consultants'
    """
    result = ''.join(i for i in s if not i.isdigit())
    return result

#%%
def remove_list_enumeration(s):
    '''
    This is a specific requirement of the NOS that comes from
    the presence of lists enumerated by strings like K+number
    or P+number. Therefore, after "lowerising" and removing 
    digits, I look for and remove strings like "k " and "p "
    '''
    result = re.sub('( k )+',' ',s)
    result = re.sub('( p )+', ' ', result)
    # it might not be necessary if I add 'k' and 'p' to stopwords
    return result

#%%
select_punct = set('!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~') #only removed "'"
extra_chars = set('•’”“µ¾âãéˆﬁ[€™¢±ï…˜')
all_select_chars = select_punct.union(extra_chars)

select_punct_skills = set('!"#$%&\()*+,./:;<=>?@[\\]^_`{|}~') #only removed "'","-"
all_select_chars_skills = select_punct_skills.union(extra_chars).union('\'')

def replace_punctuation(s):
    """
    Takes string as input.
    Removes punctuation from a string if the character is in select_punct.
    Returns a string.
   >>> replace_punctuation('sales executives/ - london')
   'sales executives   london'
    """
    for i in set(all_select_chars): #set(select_punct):
        if i in s:
            s = s.replace(i, ' ')
    return s

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

#%%

def tidy_desc(desc):
    clean_data = desc.replace('\r\n', '').replace('\xa0', '')
    nodigits = remove_digits(clean_data.lower())
    nopunct = replace_punctuation(nodigits)
    #nopunct = remove_list_enumeration(nopunct)
    lemm = lemmatise(nopunct.split())
    return ' '.join(lemm)

def tidy_desc_with_pos(desc,pof):
    clean_data = desc.replace('\r\n', '').replace('\xa0', '')
    nodigits = remove_digits(clean_data.lower())
    nopunct = replace_punctuation(nodigits)
    # add part of speech tagging
    #nopunct = [(t,nltk.pos_tag([t])[0][1]) for t in nopunct.split()]
    nopunct = nltk.pos_tag(nopunct.split())
    nopunct = [t for t in nopunct if t[1] in pos_to_wornet_dict.keys()]
    lemm = lemmatise_pruned(nopunct, pof)
    return lemm #' '.join(lemm)

def tidy_desc_with_pos_skills(desc,pof):
    clean_data = desc.replace('\r\n', '').replace('\xa0', '').lower()
    #clean_data = remove_digits(clean_data)
    nopunct = replace_punctuation_skills(clean_data)
    # add part of speech tagging
    #nopunct = [(t,nltk.pos_tag([t])[0][1]) for t in nopunct.split()]
    nopunct = nltk.pos_tag(nopunct.split())
    nopunct = [t for t in nopunct if t[1] in pos_to_wornet_dict.keys()]
    lemm = lemmatise_pruned(nopunct, pof)
    return lemm #' '.join(lemm)

#%%
def tokenize(text):
    """
    Takes string as input.
    Returns list of tokens. The function is used as an argument for
    TfidfVectorizer.
    >>> tokenize('some job title')
    ['some', 'job', 'title']
    """
    tokens = nltk.word_tokenize(text)
    return tokens

#%%
def tokenize_asis(some_list): #, stopwords):
    """
    Takes list as input.
    Returns the list with elements converted to lower case. The function is 
    used as an argument for TfidfVectorizer.
    
    In [57]: tokenize(['Accounting', 'Microsoft Excel'])
    Out[57]: ['accounting', 'microsoft excel']
    """
    tokens = [elem.lower() for elem in some_list]# if elem.lower() not in stopwords]
    return tokens


# In[62]:


#This set of functions is useful for identifying terms with highest tf-idf weights 
#in a single document or set of documents

def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding 
        feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

#%%
def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25, sparse_output = False):
    ''' Return the top n features that on average are most important 
        amongst documents in rows
        indentified by indices in grp_ids. '''
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    if sparse_output:
        return scipy.sparse.csr_matrix(top_tfidf_feats(tfidf_means, features, top_n))
    else:
        return top_tfidf_feats(tfidf_means, features, top_n)

#%%
def all_mean_feats(Xtr, grp_ids=None, min_tfidf=0.1):
    ''' Return the average
        amongst documents in rows
        indentified by indices in grp_ids. '''
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return tfidf_means

#%%
def get_top_words_weights(desc, feature_names, vect, n = 25):
    response = vect.transform(desc)
    words = top_mean_feats(response, feature_names, grp_ids = None, top_n = n)
    return words

#%%
def get_mean_tfidf(desc, vect):
    response = vect.transform(desc)
    tfidf_values = all_mean_feats(response, grp_ids = None)
    return tfidf_values

#%%
def get_top_words(desc, feature_names, vect, n = 25):
    response = vect.transform(desc)
    words = top_mean_feats(response, feature_names, grp_ids = None, top_n = n)
    return words['feature'].values

#%%
def lemmatise_pruned(x, pofs = 'nv'):
    if pofs == 'nv':
        tags = [(t,p) for t,p in x if p[:1] in ['N','V']]
    elif pofs == 'n':
        tags = [(t,p) for t,p in x if p[:1] in ['N']]
    elif pofs == 'nj':
        tags = [(t,p) for t,p in x if p[:1] in ['N','J']]
    elif pofs == 'nvj':
        tags = [(t,p) for t,p in x if p[:1] in ['N','V','J']]    
    elif pofs == 'all':
        tags = [(t,p) for t,p in x]
    else:
        raise ValueError
    return lemmatise_with_pos(tags)


#%%
def high_similarity(test_skills, comparison_vecs, clus_names):
    sims = cosine_similarity(test_skills.reshape(1,-1), comparison_vecs)
    
    top_sims = np.argsort(sims)[:, -5:].tolist()[0]
    top_sim_vals = [sims[0, elem] for elem in top_sims]
    top_sim_clus = [clus_names[elem] for elem in top_sims]
    top_sims_res = list(zip(reversed(top_sim_clus), reversed(top_sim_vals)))
    if 'dental assistance' == top_sim_clus[0]: #np.random.randn(1)>3:
        #print(df_nos_select['NOS Title'].loc[k], new_top_terms_dict[k], top_sim_clus)
        #counter +=1
        # do manual adjustment
        top_sims_res = top_sims_res[1:]
    return top_sims_res


#%%
def highest_similarity(test_skills, comparison_vecs, clus_names):
    sims = cosine_similarity(test_skills.reshape(1,-1), comparison_vecs)
    
    top_sims = np.argsort(sims)[:, -1:].tolist()[0][0] #argsort is in increasing order
    top_sim_vals = sims[0, top_sims] #[sims[0, elem] for elem in top_sims]
    top_sim_clus = clus_names[top_sims] #[clus_names[elem] for elem in top_sims]
    return top_sim_clus

#%%
def highest_similarity_top_two(test_skills, comparison_vecs, clus_names):
    sims = cosine_similarity(test_skills.reshape(1,-1), comparison_vecs)
    
    top_sims = np.argsort(sims)[:, -2:].tolist()[0]
    top_sim_vals = [sims[0, elem] for elem in top_sims]
    top_sim_clus = [clus_names[elem] for elem in top_sims]
    pct_increase = (top_sim_vals[1] - top_sim_vals[0])/top_sim_vals[0]*100
    
    if pct_increase<1:
        return top_sim_clus[::-1] #reverse because the top one is the last
    else:
        return top_sim_clus[1:]
    
#%%
def highest_similarity_threshold(test_skills, comparison_vecs, clus_names, th = 0.7):
    sims = cosine_similarity(test_skills.reshape(1,-1), comparison_vecs)
    valid_sims = sims>th
    sims = sims[valid_sims].reshape(1,-1)
    clus_names = [t for i,t in enumerate(clus_names) if valid_sims[0][i]]
    top_sims = np.argsort(sims)[:, -1:].tolist()[0]
    top_sim_vals = [sims[0, elem] for elem in top_sims]
    top_sim_clus = [clus_names[elem] for elem in top_sims]
    return top_sim_clus

#%%
def highest_similarity_threshold_top_two(test_skills, comparison_vecs, clus_names, th = 0.7):
    sims = cosine_similarity(test_skills.reshape(1,-1), comparison_vecs)
    valid_sims = sims>th
    sims = sims[valid_sims].reshape(1,-1)
    clus_names = [t for i,t in enumerate(clus_names) if valid_sims[0][i]]
    if sims.size>1:
        top_sims = np.argsort(sims)[:, -2:].tolist()[0]
    else:
        top_sims = np.argsort(sims)[:, -1:].tolist()[0]
    top_sim_vals = [sims[0, elem] for elem in top_sims]
    top_sim_clus = [clus_names[elem] for elem in top_sims]
    if len(top_sim_vals)>1:
        pct_increase = (top_sim_vals[1] - top_sim_vals[0])/top_sim_vals[0]*100
    else:
        return top_sim_clus
    if pct_increase<1:
        return top_sim_clus[::-1] #reverse because the top one is the last
    else:
        return top_sim_clus[1:]

#%%
from fuzzywuzzy import fuzz, process
def highest_fuzzymatch(skill,skill_cluster_membership,th=80,limit=3):
    out = {}
    for k in list(skill_cluster_membership.keys()):
        out[k]={}
        out[k]['results']=process.extractBests(skill,skill_cluster_membership[k],
           limit=limit, cutoff=90, scorer =fuzz.ratio)
        out[k]['mean'] = np.mean([t[1] for t in out[k]['results']])
    out_df = pd.DataFrame.from_dict(out,orient='index')
    best_cluster = out_df['mean'].idxmax()
    if out_df['mean'].loc[best_cluster]>th:
        return [best_cluster]
    else:
        return []
    