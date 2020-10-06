#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 14:51:29 2019

@author: stefgarasto
"""

#%% imports
from utils_general import *
import itertools
import numpy as np
import nltk
import scipy
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
#from fuzzywuzzy import fuzz, process
import string
import sys

# Change to where the nesta_occupation_coder github repo has been downloaded
# repo can be cloned as:
# git clone --single-branch --branch version_for_sharing https://github.com/nestauk/nesta-occupation-coder.git
title_cleaning_dir = '/Users/stefgarasto/Local-Data/scripts/nesta-occupation-coder'
try:
    if title_cleaning_dir not in sys.path:
        sys.path.append(title_cleaning_dir)
    from nesta_occupation_coder.utils.title_cleaning_utils import lemmatise, \
        remove_digits, replace_word, replace_unknown, lookup_replacement
except:
    print('Functions not found: need to clone github repo nest_occupation_coder')

######################
# Helpers
######################

#%% # useful dictionary to work with parts of speech
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

# Get punctuation list
def_punctuation = set(string.punctuation)
# remove backslash \ from the list of punctuation to remove
select_punct = def_punctuation-set('\'') #only removed "'"
# add other characters to remove - this list is manually curated
extra_chars = set('•’”“µ¾âãéˆﬁ[€™¢±ï…˜¬')
all_select_chars = select_punct.union(extra_chars)

# separate set of punctuation when dealing with skills
select_punct_skills = def_punctuation-set('-+#&')
all_select_chars_skills = select_punct_skills.union(extra_chars)#.union('\'')


################################
#A few functions for tyding up text
################################

def tag_for_lemmatise(s):
    """
    Takes token as input and returns part of speech tag compatible for use with
    lemmatisation
    """
    try:
        return pos_to_wornet_dict[nltk.pos_tag([s])[0][1]]
    except:
        return 'n'

#%%

def lemmatise_with_pos(title_terms):
    """
    Takes list as input.
    Removes suffixes if the new words exists in the nltk dictionary.
    The lemmatisation takes into account the part of speech tag of the token being
    lemmatised.
    The purpose of the function is to convert plural forms into singular.
    Allows some nouns to remain in plural form (the to_keep_asis is manually curated).
    Returns a list.
    >>> lemmatise([('teachers','NN')])
    ['teacher']
    >>> lemmatise([('analytics','NN')])
    ['analytics']
    """

    keep_asis = ['sales', 'years', 'goods', 'operations', 'systems',
                    'communications', 'events', 'loans', 'grounds',
                    'lettings', 'claims', 'accounts', 'relations',
                    'complaints', 'services']
    wnl = nltk.WordNetLemmatizer()
    processed_terms = [wnl.lemmatize(i, pos_to_wornet_dict[p]) if i not in keep_asis
                       else i for i,p in title_terms]
    return processed_terms

#%%
def lemmatise_pruned(x, pof = 'nv'):
    """
    Prepares a list of token+part-of-speech tags for lemmatisation. It only keeps
    and lemmatises those tokens who have been tagged with selected parts of speech
    Inputs:
    x - a list of tuples: each tuple is a token and its part-of-speech tag
    pof - which tags (parts of speech) to retain.

    Output:
    - lemmatised terms as a list
    """
    if pof == 'nv':
        tags = [(t,p) for t,p in x if p[:1] in ['N','V']]
    elif pof == 'n':
        tags = [(t,p) for t,p in x if p[:1] in ['N']]
    elif pof == 'nj':
        tags = [(t,p) for t,p in x if p[:1] in ['N','J']]
    elif pof == 'nvj':
        tags = [(t,p) for t,p in x if p[:1] in ['N','V','J']]
    elif pof in ['all','ignore']:
        tags = [(t,p) for t,p in x]
    else:
        raise ValueError
    return lemmatise_with_pos(tags)

#%%
def stem_features(s, ps):
    """ Stem a single token.
    Inputs:
    - s is a string
    - ps is an instance of PorterStemmer from nltk
    Outputs:
    - stemmed string
    """
    return ps.stem(s)

#%%
def remove_list_enumeration(s):
    '''
    This is a specific requirement for National Occupational Standard
    documents (project with Skills Development Scotland) that comes from
    the presence of lists enumerated by strings like K+number
    or P+number. Therefore, after "lowerising" and removing
    digits, I look for and remove strings that are only a collection of
    "k "s and "p "s

    Input and output are strings
    '''
    result = re.sub('( k )+',' ',s)
    result = re.sub('( p )+', ' ', result)
    # it might not be necessary if I add 'k' and 'p' to stopwords
    return result

#%%

def replace_punctuation_skills(s, SKILLS = False):
    """
    Takes string as input.
    Removes punctuation from a string if the character is in select_punct.
    Returns a string.
   >>> replace_punctuation('sales executives/ - london')
   'sales executives   london'
    """
    if SKILLS:
        use_select_chars = all_select_chars_skills
    else:
        use_select_chars = all_select_chars
    assert(isinstance(use_select_chars,set))
    for i in use_select_chars: #set(select_punct):
        if i in s:
            s = s.replace(i, ' ')
    # last thing: if there are spurious dashes, then remove them:
    s = s.replace(' - ',' ')
    # remove multiple consecutive white spaces
    return s

#%%
# remove white spaces
def remove_white_spaces(my_string):
    return ' '.join(my_string.split())

#%%
def apply_lower(my_string):
    return my_string.lower()

#%%
def tidy_desc(desc,pof,DIGITS=True,SKILLS=False):
    '''
    This function combines various steps
    '''
    clean_data = desc.replace('\r\n', '').replace('\xa0', '')
    if DIGITS:
        # don't remove digits
        nodigits = clean_data.lower()
    else:
        nodigits = remove_digits(clean_data.lower())
    nopunct = replace_punctuation_skills(nodigits, SKILLS=SKILLS)
    # add part of speech tagging
    if pof != 'ignore':
        nopunct = nltk.pos_tag(nopunct.split())
        nopunct = [t for t in nopunct if t[1] in pos_to_wornet_dict.keys()]
        lemm = lemmatise_pruned(nopunct, pof)
        return lemm
    else:
        lemm = lemmatise(nopunct.split())
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


#%%
def get_mean_vec(skill_list, model, weights= None):
    """
    Get (weighted) average embedding for a list of skills (skill_list)
    Model is a Word2Vec model used to compute the embeddings

    Returns the average embedding and the full matrix of skill embeddings (with
    dimenstions #skills x #features)
    """
    if not weights:
        weights = np.ones(len(skill_list))
    skill_list_conv = skill_list
    vector_list = [sentence_to_vectors_nofile(elem,model)[0]*weights[ix] for
                    ix,elem in enumerate(skill_list_conv)]
    wvector_list = [elem*weights[ix] for ix,elem in enumerate(vector_list)]
    vec_array = np.asarray(vector_list)
    wvec_array = np.asarray(wvector_list)
    # average across skills
    avg_vec = np.mean(wvec_array, axis=0)
    return avg_vec, vec_array


#%%
def get_average_skill_category(list_of_skills, reference_dict):
    """
    Returns top 10 categories in a feature array averaged across "skills".

    Inputs:
    list_of_skills: list of skills to consider
    reference_dict: dictionary with skills as keys and corresponding feature vectors
    as values (one feature vector per skill)

    Outpus:
    list of tuples with top 10 categories (as array indices) and corresponding values
    """
    pruned_skills = [elem for elem in list_of_skills if elem in reference_dict]
    if len(pruned_skills):
        # vec_list dimensions are #skills x #features
        vec_list = [reference_dict[skill] for skill in pruned_skills]
        vec_array = np.asarray(vec_list)
        # average across skills
        avg_vec = np.mean(vec_array, axis=0)
        # select top 10 values
        sorted_vecs = np.argsort(avg_vec)[0, -10:]
        scores = [avg_vec[0,i] for i in sorted_vecs]
        categories_values = zip(sorted_vecs, scores)
        res = list(categories_values)
    else:
        res = []
    return res

#%%
def get_best_skill_category(list_of_skills, reference_dict, transversal):
    """ Selects a subset of the top 10 features in an array averages across skills
    by discarding those features that are in "transversal" (intended as indices
    for an array).
    """
    top10 = get_average_skill_category(list_of_skills, reference_dict)
    if len(top10):
        # discard features indices that are in transversal
        dom_specific = [elem for elem in top10 if elem[0] not in transversal]
        # take the feature with the highest score
        best = max(dom_specific, key = lambda x: x[1])
    else:
        best = (999, 0.0)
    return best

#%%

##################
#This set of functions is useful for identifying terms with highest tf-idf weights
#in a single document or set of documents
##################

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
        amongst documents in Xtr indentified by indices in grp_ids.
        Returns a sparse matrix or a dataframe with feature names and values '''
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
    ''' Return the average feature vectors
        amongst documents in Xtr
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
    """
    Transform a document corpus (desc) into TF-IDF vectors (via "vect") and take
    the top n features, on average across documents.
    Returns both the feature names and values
    """
    response = vect.transform(desc)
    words = top_mean_feats(response, feature_names, grp_ids = None, top_n = n)
    return words

#%%
def get_mean_tfidf(desc, vect):
    """
    Transform a document corpus (desc) into TF-IDF vectors (via "vect") and take
    the average across documents
    """
    response = vect.transform(desc)
    tfidf_values = all_mean_feats(response, grp_ids = None)
    return tfidf_values

#%%
def get_top_words(desc, feature_names, vect, n = 25):
    """
    Transform a document corpus (desc) into TF-IDF vectors (via "vect") and take
    the top n features, on average across documents.
    Returns only the feature names
    """
    response = vect.transform(desc)
    words = top_mean_feats(response, feature_names, grp_ids = None, top_n = n)
    return words['feature'].values


############################
# Group of functions to identify skill cluster(s) with highest semantic similarity
# to target skill or word. There are several variations on this theme.
# I made a different function for each case to avoid a proliferation of if/else
# based on input arguments, but it's not necessarily the best course of action
############################
#%%
def high_similarity(test_skills, comparison_vecs, clus_names):
    """
    Return top 5 skill clusters with the highest semantic similarity with the target
    skill. Semantic similarity is defined as the cosine similarity of the
    word (or sentence) embeddings.

    Inputs:
    test_skills: single embedding vector for the target skill
    comparison_vecs: one vector per skill clusters (dimensions: #clusters x #features)
    clus_names: names of the skill clusters

    Outputs:
    top_sims_res: names of top 5 clusters and corresponding similarity values
    """
    sims = cosine_similarity(test_skills.reshape(1,-1), comparison_vecs)

    top_sims = np.argsort(sims)[:, -5:].tolist()[0]
    top_sim_vals = [sims[0, elem] for elem in top_sims]
    top_sim_clus = [clus_names[elem] for elem in top_sims]
    top_sims_res = list(zip(reversed(top_sim_clus), reversed(top_sim_vals)))
    if 'dental assistance' == top_sim_clus[0][0]:
        # This is an ad-hoc correction when working with National Occupational
        # Standards: won't influence any result
        top_sims_res = top_sims_res[1:]
    return top_sims_res


#%%
def highest_similarity(test_skills, comparison_vecs, clus_names):
    """
    Return the skill cluster with the highest semantic similarity with the target
    skill. Semantic similarity is defined as the cosine similarity of the
    word (or sentence) embeddings.

    Inputs:
    test_skills: single embedding vector for the target skill
    comparison_vecs: one vector per skill clusters (dimensions: #clusters x #features)
    clus_names: names of the skill clusters

    Outputs:
    top_sim_clus: top matching cluster name
    """
    sims = cosine_similarity(test_skills.reshape(1,-1), comparison_vecs)

    top_sims = np.argsort(sims)[:, -1:].tolist()[0][0] #argsort is in increasing order
    top_sim_vals = sims[0, top_sims]
    top_sim_clus = clus_names[top_sims]
    return top_sim_clus

#%%
def highest_similarity_top_two(test_skills, comparison_vecs, clus_names):
    """
    Return the top 2 (at most) skill clusters with the highest semantic similarity
    with the target skill. Semantic similarity is defined as the cosine similarity
    of the word (or sentence) embeddings. It returns the top 2 cluster if the
    relative difference in similarity is less than 1%, otherwise only returns the
    top match.

    Inputs:
    test_skills: single embedding vector for the target skill
    comparison_vecs: one vector per skill clusters (dimensions: #clusters x #features)
    clus_names: names of the skill clusters

    Outputs:
    top_sim_clus: top matching cluster name(s)
    """
    sims = cosine_similarity(test_skills.reshape(1,-1), comparison_vecs)

    top_sims = np.argsort(sims)[:, -2:].tolist()[0]
    top_sim_vals = [sims[0, elem] for elem in top_sims]
    top_sim_clus = [clus_names[elem] for elem in top_sims]
    # compute similarity increase in percentage between cluster nb 1 and 2
    pct_increase = (top_sim_vals[1] - top_sim_vals[0])/top_sim_vals[0]*100
    # return top or top 2 cluster(s) based on relative difference in similarity scores
    if pct_increase<1:
        return top_sim_clus[::-1] #reverse because the top one is the last
    else:
        return top_sim_clus[1:]

#%%
def highest_similarity_threshold(test_skills, comparison_vecs, clus_names, th = 0.7):
    """
    Return the top skill cluster with the highest semantic similarity
    with the target skill. Semantic similarity is defined as the cosine similarity
    of the word (or sentence) embeddings. It only returns the top cluster if the
    similarity score is higher than a certain threshold.

    Inputs:
    test_skills: single embedding vector for the target skill
    comparison_vecs: one vector per skill clusters (dimensions: #clusters x #features)
    clus_names: names of the skill clusters
    th: threshold similarity value

    Outputs:
    top_sim_clus: top matching cluster name(s)
    """
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
    """
    Return the top 2 (at most) skill clusters with the highest semantic similarity
    with the target skill. Semantic similarity is defined as the cosine similarity
    of the word (or sentence) embeddings. It returns the top 2 cluster if the
    relative difference in similarity is less than 1%, otherwise only returns the
    top match. However, either cluster is only returned as output if its
    similarity score is higher than a certain threshold.
    th: threshold similarity value

    Inputs:
    test_skills: single embedding vector for the target skill
    comparison_vecs: one vector per skill clusters (dimensions: #clusters x #features)
    clus_names: names of the skill clusters

    Outputs:
    top_sim_clus: top matching cluster name(s)
    """
    sims = cosine_similarity(test_skills.reshape(1,-1), comparison_vecs)
    # only keep those clusters with similarity score higher than a certain threshold
    valid_sims = sims>th
    sims = sims[valid_sims].reshape(1,-1)
    clus_names = [t for i,t in enumerate(clus_names) if valid_sims[0][i]]
    if sims.size>1:
        top_sims = np.argsort(sims)[:, -2:].tolist()[0]
    else:
        top_sims = np.argsort(sims)[:, -1:].tolist()[0]
    top_sim_vals = [sims[0, elem] for elem in top_sims]
    top_sim_clus = [clus_names[elem] for elem in top_sims]
    # compute relative difference between top 2 clusters
    if len(top_sim_vals)>1:
        pct_increase = (top_sim_vals[1] - top_sim_vals[0])/top_sim_vals[0]*100
    else:
        return top_sim_clus
    if pct_increase<1:
        return top_sim_clus[::-1] #reverse because the top one is the last
    else:
        return top_sim_clus[1:]

#%%
def highest_fuzzymatch(skill,skill_cluster_membership,th=80,limit=3):
    """
    Given a target skill, returns the skill cluster with the highest match based
    on the fuzzy matches between the target skill (skill) and the skills
    comprising the cluster. The final matching score is taken as the average of
    the top n fuzzy matching scores (n = limit) between the target skill and the skills
    comprising each cluster. The top cluster is returned only if this average is
    higher than a specific threshold (th).

    Note: I am not particularly happy with how well this method performs.
    """
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
