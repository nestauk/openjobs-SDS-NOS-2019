#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:21:06 2019

@author: stefgarasto
"""

import numpy as np
import pandas as pd
import time
from utils_bg import *
from utils_nlp import *
import gensim
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn import mixture
import re
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from fuzzywuzzy import process
import pickle
from collections import Counter
from utils_nlp import flatten_lol

#%% some parameters are needed here
output_dir = '/Users/stefgarasto/Google Drive/Documents/results/NOS/progression_pathways/joined_LSH/'
#output_dir = '/Users/stefgarasto/Google Drive/Documents/results/NOS/nlp_analysis/'

# In[13]:

lookup_dir= '/Users/stefgarasto/Google Drive/Documents/results/NOS/extracted/'
lookup_dir2= '/Users/stefgarasto/Google Drive/Documents/results/NOS/nlp_analysis/'

all_edu = ['Pregraduate','Graduate','Postgraduate']
all_exp =  ['Entry-level','Mid-level','Senior-level']

edu_colours = {'Pregraduate': nesta_colours[0], 'Graduate': nesta_colours[1],
                       'Postgraduate': nesta_colours[3]}
exp_colours = {'Entry-level': nesta_colours[4], 'Mid-level': nesta_colours[6],
                       'Senior-level': nesta_colours[8]}


# In[17]:
''' Function definitions '''

# manually remove "k"s and "p"s from the pruned columns
def remove_pk(x):
    return [t for t in x if t not in ['k','p']]

#%%
        
def assign_supersuite(x, all_match_names):
    for supersuite in all_match_names.keys():
        if x in all_match_names[supersuite]:
            return supersuite.lower()
    # if no match has been found
    return 'other'

def adjustsoccode(x):
    y = re.findall(r"[\d']+", str(x))
    if len(y):
        return y[0][1:-1]
    else:
        return np.nan

def extract2digits(x):
    if isinstance(x,str):
        try:
            return float(x[:2])
        except:
            return np.nan
    else:
        return np.nan
    
def extract3digits(x):
    if isinstance(x,str):
        try:
            return float(x[:3])
        except:
            return np.nan
    else:
        return np.nan
    
def extract1digits(x):
    if isinstance(x,str):
        try:
            return float(x[:1])
        except:
            return np.nan
    else:
        return np.nan

def extract4digits(x):
    if isinstance(x,str):
        try:
            return float(x)
        except:
            return np.nan
    else:
        return np.nan


def define_tfidf(params, stopwords):
    if params['ngrams'] == 'bi':
        tfidf = TfidfVectorizer(tokenizer=tokenize_asis,
                                lowercase = False,
                                stop_words=stopwords,
                                ngram_range=(1,2), 
                                max_df = params['tfidf_max'], 
                                min_df = params['tfidf_min'])
    elif params['ngrams'] == 'tri':
        tfidf = TfidfVectorizer(tokenizer=tokenize_asis,
                                lowercase = False,
                                stop_words=stopwords,
                                ngram_range=(1,3), 
                                max_df = params['tfidf_max'], 
                                min_df = params['tfidf_min'])
    else:
        # unigrams is the default
        #lambda x: tokenize_asis(x,stopwords),
        tfidf = TfidfVectorizer(tokenizer=tokenize_asis, 
                                lowercase = False,
                                stop_words=stopwords,
                                max_df = params['tfidf_max'], 
                                min_df = params['tfidf_min'])
    return tfidf


# In[40]:


# now, collect the text to transform
def combine_nos_text(df_nos_select, col = 'pruned'):
    all_joint_tokens = []
    # group by suites and concatenate all docs in it
    row_names = []
    for name, group in df_nos_select.groupby('One_suite'):
        row_names.append(name)
        joint_tokens = []
        for idoc in group[col].index:
            joint_tokens += group[col].loc[idoc]
        all_joint_tokens.append(joint_tokens)
    # return a dataframe
    return pd.DataFrame({'tokens': all_joint_tokens}, index = row_names)

def get_tfidf_matrix(params, df_nos_select, tfidf, col = 'pruned'):
    # Note: this can simply be used to get the tfidf transform, by setting 
    # bywhich=docs and any mode
    t0 = time.time()
    # first, get the dataframe of tokens
    if params['bywhich'] == 'docs':
        textfortoken = df_nos_select[col]
        
    elif params['bywhich'] == 'suites':
        if params['mode'] == 'meantfidf':
            textfortoken = df_nos_select[col]
                
        elif params['mode'] == 'combinedtfidf':
            # note that this is the only case where the tfidf min and max are 
            # computed considering the number of 
            # suites as the number of elements in the collection.
            # TODO: allow for the alternative case, where the transform is computed 
            # on individual NOS and then 
            # applied to the joint tokens
            textfortoken = combine_nos_text(df_nos_select, col = col)['tokens']
    
    # apply tfidf transform to the tokenised text
    tfidfm = tfidf.fit_transform(textfortoken)
    feature_names = tfidf.get_feature_names()
    
    # if the average is needed, compute it and overwrite the matrix. Note that the 
    # step above is still needed to
    # initialise the tfidf transform with the proper features and stopwords
    if (params['bywhich'] == 'suites') and (params['mode'] =='meantfidf'):
        row_names = df_nos_select['One_suite'].value_counts().index.values
        tfidfm = scipy.sparse.lil_matrix(np.zeros((len(row_names),
                                        len(feature_names)), dtype = np.float32))
        igroup = 0
        for name, group in df_nos_select.groupby('One_suite'):
            tmp = get_mean_tfidf(group[col], tfidf)
            tfidfm[igroup] = tmp
            igroup+=1

    print_elapsed(t0, 'computing the tfidf matrix')
    return tfidfm, feature_names, tfidf, textfortoken


# In[41]:


def get_top_keywords(df, name, stopwords, top_n = 20):
    all_keywords = []
    count_keywords = {}
    for ix in df.index:
        if isinstance(df.loc[ix], list):
            for ik in df.loc[ix]:
                # I think that ik can be a collection of words separated by ";"
                #ik_elems = ik.split(';')
                ik_elems = re.findall(r"[\w']+", ik.replace('-',''))
                # remove extra spaces
                ik_elems = [elem.strip() for elem in ik_elems]
                # remove digits
                ik_elems = [elem for elem in ik_elems if not elem.isdigit()]
                for elem in ik_elems:
                    if elem not in stopwords:
                        if elem not in all_keywords:
                            all_keywords.append(elem)
                            count_keywords[elem] = 1
                        else:
                            count_keywords[elem] += 1
        elif isinstance(df.loc[ix],str):
            ik_elems = re.findall(r"[\w']+", df.loc[ix].replace('-',''))
            #ik_elems = re.split('; |, ', df.loc[ix])
            # remove extra spaces
            ik_elems = [elem.strip() for elem in ik_elems]
            # remove digits
            ik_elems = [elem for elem in ik_elems if not elem.isdigit()]
            for elem in ik_elems:
                if elem not in stopwords:
                    if elem not in all_keywords:
                        all_keywords.append(elem)
                        count_keywords[elem] = 1
                    else:
                        count_keywords[elem] += 1
    n_repeated = np.sum(np.array(list(count_keywords.values()))>1)
    n_keywords = len(all_keywords)
    #print('Number of keywords repeated more than once for suite {} is {}. \n'.format(name,
    #                                                            n_repeated))
    # get the top 20 keywords in terms of count
    top_kw_indices = np.argsort(list(count_keywords.values()))[::-1][:top_n]
    top_keywords = [k for t,k in enumerate(all_keywords) if t in top_kw_indices]
    for _ in range(len(top_keywords),top_n):
        top_keywords.append('-')
    return top_keywords, n_keywords, n_repeated

def get_top_keywords_nos(nos,stopwords, top_n = 20):
    all_keywords = []
    count_keywords = {}
    if isinstance(nos, list):
        for ik in nos:
            # I think that ik can be a collection of words separated by ";"
            #ik_elems = ik.split(';')
            ik_elems = re.findall(r"[\w']+", ik.replace('-',''))
            # remove extra spaces
            ik_elems = [elem.strip() for elem in ik_elems]
            # remove digits
            ik_elems = [elem for elem in ik_elems if not elem.isdigit()]
            for elem in ik_elems:
                if elem not in stopwords:
                    if elem not in all_keywords:
                        all_keywords.append(elem)
                        count_keywords[elem] = 1
                    else:
                        count_keywords[elem] += 1
    elif isinstance(nos,str):
        ik_elems = re.findall(r"[\w']+", nos.replace('-',''))
        #ik_elems = re.split('; |, ', nos)
        # remove extra spaces
        ik_elems = [elem.strip() for elem in ik_elems]
        # remove digits
        ik_elems = [elem for elem in ik_elems if not elem.isdigit()]
        for elem in ik_elems:
            if elem not in stopwords:
                if elem not in all_keywords:
                    all_keywords.append(elem)
                    count_keywords[elem] = 1
                else:
                    count_keywords[elem] += 1
    n_repeated = np.sum(np.array(list(count_keywords.values()))>1)
    n_keywords = len(all_keywords)
    #print('Number of keywords repeated more than once for suite {} is {}. \n'.format(name,
    #                                                            n_repeated))
    # get the top 20 keywords in terms of count
    top_kw_indices = np.argsort(list(count_keywords.values()))[::-1][:top_n]
    top_keywords = [k for t,k in enumerate(all_keywords) if t in top_kw_indices]
    for _ in range(len(top_keywords),top_n):
        top_keywords.append('-')
    return top_keywords, n_keywords, n_repeated
#df_nos.sample(n=3)

#%%

def select_subdf(SELECT_MODE, clusters2use, nos_clusters, df_nos_select):
    if isinstance(SELECT_MODE, str):
        tmp_dict = {'engineering': 'Engineering', 'management': 'Management',
                    'financialservices': 'Financial services', 
                    'construction': 'Construction'}
        # select NOS from super suite
        cluster_name = SELECT_MODE
        cluster_name_save = cluster_name
        cluster_name_figs = tmp_dict[SELECT_MODE]
        subset_nos = df_nos_select[df_nos_select['supersuite'].map(
                lambda x: SELECT_MODE in x)]
    elif isinstance(SELECT_MODE, int):
        cluster_name = clusters2use[SELECT_MODE][1]
        cluster_name_save = cluster_name.replace(' ','_')
        cluster_name_figs = cluster_name.capitalize()
        suites2use = list(nos_clusters[nos_clusters['hierarchical'].map(
                lambda x: x in clusters2use[SELECT_MODE][0])]['Suite_names'].values)
        subset_nos = df_nos_select[df_nos_select['One_suite'].map(
                lambda x: x in suites2use)]
    elif isinstance(SELECT_MODE, pd.core.series.Series):
        cluster_name = SELECT_MODE.name
        cluster_name_save = cluster_name.lower().replace(' ','_')
        cluster_name_figs = cluster_name
        indices2take = nos_clusters[
                nos_clusters['labels']==SELECT_MODE['Cluster index']]['index']
        subset_nos = df_nos_select.loc[indices2take]
    #    
    print('Number of NOS selected: ', len(subset_nos))
    #print(subset_nos.columns)
    #%
    # only select those engineering nos with SOC codes
    nosoc = subset_nos['SOC4'].isnull()
    print('percentage of nos without SOC codes: ', nosoc.sum()/len(nosoc))
    if (nosoc.sum())/len(nosoc)<0.9:
        final_nos = subset_nos[~nosoc] #np.isnan(engineering_nos['SOC4'])]
    else:
        final_nos = subset_nos
    final_groups = final_nos.groupby(by = 'One_suite')
    larger_suites = []
    all_lengths = final_groups.agg(len)['NOS Title'].values
    all_names = final_groups.groups.keys()
    # remove joined up supersuites (legacy of joining LSH duplicates)
    goodnames = [(';' not in t) and (all_lengths[ix]>2) for ix,t in enumerate(all_names)]
    all_names = [t for ix,t in enumerate(all_names) if goodnames[ix]]
    all_lengths = np.array([t for ix,t in enumerate(all_lengths) if goodnames[ix]])
    
    if not isinstance(SELECT_MODE, str):
        larger_suites = all_names
    else:    
        # get the top 15
        idxsort = np.argsort(all_lengths)
        all_lengths = [all_lengths[ix] for ix in idxsort[::-1]]
        larger_suites = [all_names[ix] for ix in idxsort[::-1]]
        larger_suites = larger_suites[:15]
    #all_lengths[::-1].sort()
    print(all_lengths)
    ##th_supers = ['engineering': 40, 'financialservices': ]
    #for name, group in final_groups:
    #    all_names.append(name)
    #for name, group in final_groups:
    #    if not isinstance(SELECT_MODE, str):
    #        larger_suites.append(name)
    #    elif len(group)> all_lengths[15]:#th_supers[SELECT_MODE]:
    #        #print(name, len(group))
    #        larger_suites.append(name)
     #       
    return final_nos, final_groups, larger_suites, cluster_name,  \
                    cluster_name_save, cluster_name_figs

#%%
def fuzzy_match_to_list(x, comparison_list, match_th = 90):
    y = x.split(';')
    if len(y)>1:
        # recursion, for each substring return the max similarity with a flattened version
        # of the original comparison list
        new_comparison_list = flatten_lol([t.split(';') for t in comparison_list])
        all_matches = [fuzzy_match_to_list(sub_x, new_comparison_list) 
                            for sub_x in x.split(';')]
        # return True if all substrings have at least one very good match
        return all(all_matches) #[t>match_th for t in all_matches])
    else:
        all_matches = process.extract(x, comparison_list)
        all_matches = [t[1] for t in all_matches]
        # return True if there is at least one very good match
        return np.max(all_matches)>match_th

def select_subdf_from_list(SELECT_MODE, clusters2use, nos_clusters, df_nos_select):
    assert(isinstance(SELECT_MODE,pd.core.series.Series))
    flags = nos_clusters.labels.map(lambda x: x in SELECT_MODE['Cluster index'])
    indices2take = nos_clusters[flags]['index']
    #subset_nos = df_nos_select.loc[indices2take]
    subset_nos = df_nos_select[df_nos_select.index.map(lambda x: x in list(indices2take))]
    # eliminate potential NaNs
    subset_nos = subset_nos[~subset_nos['NOS Title'].isnull()]
    print('Number of NOS selected: ', len(subset_nos))
    
    # I might have deleted some NOS manually
    selected_columns =SELECT_MODE.index[SELECT_MODE.index.map(lambda x:isinstance(x,int))]
    list_of_titles = SELECT_MODE.loc[selected_columns]
    list_of_titles = list(list_of_titles[~list_of_titles.isnull()])
    if len(list_of_titles) < len(subset_nos):
        flags = subset_nos['NOS Title'].map(lambda x: fuzzy_match_to_list(x,list_of_titles,
                      match_th = 90))
        #.map(lambda x: x in list_of_titles)
        subset_nos = subset_nos[flags]
    
    cluster_name = SELECT_MODE.name
    cluster_name_save = cluster_name.lower().replace(' ','_')
    cluster_name_figs = cluster_name
    
    print('Number of NOS selected: ', len(subset_nos))
    #%
    # only select those engineering nos with SOC codes
    nosoc = subset_nos['SOC4'].isnull()
    print('percentage of nos without SOC codes: ', nosoc.sum()/len(nosoc))
    if (nosoc.sum())/len(nosoc)<0.9:
        final_nos = subset_nos[~nosoc] #np.isnan(engineering_nos['SOC4'])]
    else:
        final_nos = subset_nos
    final_groups = final_nos.groupby(by = 'One_suite')
    larger_suites = []
    all_lengths = final_groups.agg(len)['NOS Title'].values
    all_lengths[::-1].sort()
    print(all_lengths)
    #th_supers = ['engineering': 40, 'financialservices': ]
    for name, group in final_groups:
        if not isinstance(SELECT_MODE, str):
            larger_suites.append(name)
        elif len(group)> all_lengths[15]:#th_supers[SELECT_MODE]:
            #print(name, len(group))
            larger_suites.append(name)
     #       
    return final_nos, final_groups, larger_suites, cluster_name,  \
                    cluster_name_save, cluster_name_figs
                    
#%%                   
def replace_oob_socs(x):
    if isinstance(x, list):
        for ix,x_value in enumerate(x):
            if x_value in matches_oobsoc_to_soc2:
                x[ix] = matches_oobsoc_to_soc2[x_value]
    else:
        if not np.isnan(x):
            if x in matches_oobsoc_to_soc2:
                x = matches_oobsoc_to_soc2[x]
        else:
            return x
    return x
        
#%%
def extract_top_skills(final_nos,all_exp,all_edu,N=20):
    skillsdf = []
    #N = 20
    for exp0 in all_exp:
        for edu0 in all_edu:
            tmp = final_nos[(final_nos['myExp-peak']==exp0) & 
                            (final_nos['myEdu-peak']==edu0)]
            if len(tmp)>0:
                tmp = tmp['converted_skills'].agg(sum)
                Ntot =sum([t for t in tmp.values()])
                print(exp0,edu0,Ntot,max([t for t in tmp.values()]))#,np.std(tmp.values()))
                # only keep the N most important skills
                A = tmp.most_common()
                A = A[:N]
                tmpdf = pd.DataFrame([(key, np.around(t/Ntot,3)) for key,t in A])
                tmpdf['Qualification requirements'] = edu0
                tmpdf['Experience requirements'] = exp0
                tmpdf['skill rank'] = np.arange(1,len(tmpdf)+1)
                skillsdf.append(tmpdf)
                #break
    skillsdf = pd.concat(skillsdf)
    skillsdf = skillsdf.set_index(['Qualification requirements',
                                        'Experience requirements'])
    skillsdf = skillsdf.rename(columns = {0: 'skill', 
                                          1: 'occurrence percentage'})
    return skillsdf

#%%
def plot_centroids_hm(final_small,w,h, cent_col = 'centEdu',qual_col = all_edu,
                      xcat = 'Education category', cluster_name_figs = '',
                      cluster_name_save = '', KEY='', output_dir = output_dir,
                      SAVEFIG = False):
    if len(final_small)<=6:
        h = h+2
    fig = plt.figure(figsize = (w,h))
    sns.heatmap(final_small.sort_values(cent_col)[qual_col].values, annot = True,
                linewidths =0.2)
    t=plt.yticks(ticks = .5 + np.arange(len(final_small)), 
                 labels = final_small.sort_values(cent_col)['NOS Title'].map(
                         lambda x: x.capitalize()), fontsize = 20, 
                         rotation = 0,
                     rotation_mode="anchor")
    ## colour the labels according to the other category
    #T = plt.yticks()
    #for t in T[1]:
    #    k = final_small[final_small['NOS Title']== t.get_text().lower()][other_col]
    #    t.set_color(exp_colours[k.values[0]])
    # # note: other_col would be myEdu-peak or myExp-peak
    plt.xticks(ticks = [0,1,2], labels = qual_col, fontsize = 20, rotation = 30)
    plt.ylabel('NOS title', fontsize  =22)
    plt.xlabel(xcat, fontsize = 22)
    xcat_red = xcat.split()[0].lower()
    plt.title('NOS ordered by {} \n for {}'.format(xcat_red,cluster_name_figs),
              fontsize = 22)
    plt.tight_layout()
    #if cluster_name_save in ['engineering','construction','management','financialservices']:
    #    output_dir += '/supersuites'
    #else:
    #    output_dir += '/nosclusters'
    if SAVEFIG:
        plt.savefig(output_dir + '/NOS_cent_{}_ordered_for_{}_{}.png'.format(
                xcat_red[:3],cluster_name_save,KEY), bbox_inches='tight')
        plt.close(fig)
        
#%%
def plot_swarm_nos(final_nos, SALARY = True, cluster_name_save = '', KEY = '',
                   title = '', output_dir = output_dir, SAVEFIG = False):
    plt.figure(figsize = (8,8))
    #SALARY = True
    #output_dir = '/Users/stefgarasto/Google Drive/Documents/results/NOS/progression_pathways/'
    if SALARY:
        #%
        f=sns.scatterplot(x='centExp',hue ='Qualification requirements', y= 'Salary-peak', 
            data= final_nos.rename(columns = {'myEdu-peak':'Qualification requirements'}),
            palette = [nesta_colours[t] for t in [0,1,3]],
            hue_order = ['Pregraduate','Graduate','Postgraduate'])
        #A = plt.gca().get_xticklabels()
        #%
        plt.xlabel('Experience centroid', fontsize = 18)
        plt.ylabel('Average salary', fontsize = 18) # how to add std bar?
        plt.title(title, fontsize = 18)
        xt = plt.xticks()
        xt = np.array(xt)[0]
        plt.xticks([1,2,3], all_exp, fontsize = 16)
#                       [xt[0],xt[0]*.5+xt[-1]*.5,xt[-1]], all_exp, fontsize = 16, 
#                       rotation = 20, ha="right",
#                         rotation_mode="anchor")
    
    else:
        f=sns.swarmplot(x='centExp',y ='centEdu', hue= 'Salary-peak', 
            data= final_nos.rename(columns = {'myEdu-peak':'Qualification requirements'}),
            palette = 'inferno_r')
        plt.xlabel('Experience centroid', fontsize = 18)
        plt.ylabel('Education centroid', fontsize = 18) # how to add std bar?
        plt.xticks([0,13,26], all_exp, fontsize = 16, rotation = 20)
        plt.yticks([1,2,3], all_edu, fontsize = 16, rotation = 20)
    #if cluster_name_save in ['engineering','construction','management','financialservices']:
    #    output_dir = output_dir + '/supersuites'
    #else:
    #    output_dir += '/nosclusters'
    if SAVEFIG:
        plt.savefig(output_dir + '/NOS_cent_edu_exp_ordered_for_{}_{}.png'.format(
                cluster_name_save,KEY), bbox_inches='tight')
        plt.close('all')

#%%
# implement K-means
def do_kmean(xx, ks = np.arange(2,4),N=100):
    stab = []
    for k in ks:
        t0 = time.time()
        # do N iterations
        stab0 = []
        A = np.empty((xx.shape[0],N))
        for i in range(N):
            k_clus = KMeans(k, n_init = 1, random_state = np.random.randint(1e7))
            A[:,i] = k_clus.fit_predict(xx)
            for j in range(i):
                stab0.append(adjusted_rand_score(A[:,i],A[:,j]))
        # get stability of clusters for this nb of clusters
        stab.append(np.mean(stab0))
        print_elapsed(t0,'kmeans for k={}'.format(k))
    # what number of clusters has highest stability?
    kmax = ks[np.array(stab).argmax()]
    # redo one last clustering with kmax 
    # and lots of iteration to get the stable versions
    k_clus = KMeans(kmax, n_init= 100)
    labels = k_clus.fit_predict(xx)
    return labels, k_clus, kmax, stab

#%%
def do_gmm(xx, ks = np.arange(2,6),N=100):
    #bgmm =mixture.BayesianGaussianMixture(xx.shape[0],'full',
    #                    max_iter = 1000,weight_concentration_prior = 1e-3)
    #labels0 = bgmm.fit_predict(xx)
    stabgmm = []
    bicgmm = []
    aicgmm = []
    for k in ks:
        t0 = time.time()
        # do 100 iterations
        stab0gmm = []
        B = np.empty((xx.shape[0],N))
        bictmp = 0
        aictmp = 0
        for i in range(N):
            gmm = mixture.GaussianMixture(k, n_init=1, 
                                    random_state = np.random.randint(1e7))
            B[:,i] = gmm.fit_predict(xx)
            bictmp += gmm.bic(xx)/N
            aictmp += gmm.aic(xx)/N
            for j in range(i):
                stab0gmm.append(adjusted_rand_score(B[:,i],B[:,j]))
        # get stability of clusters for this nb of clusters
        bicgmm.append(bictmp)
        aicgmm.append(aictmp)
        stabgmm.append(np.mean(stab0gmm))
        print_elapsed(t0,'kmeans for k={}. Stability is {:.4f}'.format(k,stabgmm[-1]))
    # what number of clusters has highest stability?
    #kmaxgmm1 = ks[np.array(stabgmm).argmax()]
    # what's the minimum number of clusters with high stability?
    try:
        kmaxgmm1 = np.where(stabgmm>.9)[0][0]+1
    except:
        #if none is more than .9 take the max
        kmaxgmm1 = ks[np.array(stabgmm).argmax()]
    kmaxgmm2 = ks[np.array(bicgmm).argmin()]
    kmaxgmm3 = ks[np.array(aicgmm).argmin()]
    # if the stability is less than 0.7 for all ks, there is only 1 cluster
    if max(stabgmm)<0.7:
        kmaxgmm = 1
    else:
        # take the minimum possible number of clusters
        kmaxgmm = min([kmaxgmm1, kmaxgmm2, kmaxgmm3])
    # redo one last clustering with kmax 
    # and lots of iteration to get the stable versions
    gmm = mixture.GaussianMixture(kmaxgmm, n_init=100, 
                                  random_state = np.random.randint(1e7))
    labelsgmm = gmm.fit_predict(xx)
    #bicgmm1 = gmm.bic(xx)
    #aicgmm1 = gmm.aic(xx)
    return labelsgmm, gmm, kmaxgmm, (stabgmm, bicgmm, aicgmm)
#%%
def do_bgmm(xx, ks = np.arange(2,4), N=100):
    # this one should find the optimal number of clusters automatically
    bgmm =mixture.BayesianGaussianMixture(xx.shape[0],'full', n_init = N,
                        max_iter = 1000, weight_concentration_prior = 1e-3)
    labels = bgmm.fit_predict(xx)
    kmax= len(np.unique(labels))
    return labels, bgmm, kmax, 0

#%
def take_prc(x,p):
    try:
        return np.percentile(x,p)
    except:
        return np.nan   

#%%
def extract_vals(df,all_edu,all_exp):
    vals = np.array(df[all_edu[:2] + all_exp[:2]],
                                        dtype =np.float32)
    #icol = vals.shape[1]
    for prc in [25,50,75]:
        zz = df['Salary'].map(lambda x: take_prc(x,prc
                        )).values
        zz[np.isnan(zz)] = np.nanmean(zz)
        vals = np.concatenate((vals,zz[:,np.newaxis]),axis = 1)
        #vals[np.isnan(vals[:,6]),6] = np.nanmean(vals[:,6])
    return vals

#%%
def extract_top_features(tfidfm_row,feature_names,N=20):
    top_ngrams = np.argsort(tfidfm_row)
    top_ngrams = top_ngrams.tolist()[0][-N:]
    # reverse the order, so that the first one is the most important
    top_ngrams = top_ngrams[::-1]
    # only retain the ones with non zero features
    top_ngrams = [elem for elem in top_ngrams if tfidfm_row[0,elem]>0]
    top_weights = [tfidfm_row[0,elem] for elem in top_ngrams]
    top_features = [feature_names[elem] for elem in top_ngrams]
    return top_ngrams, top_weights, top_features

#Loading a pre-trained glove model into gensim
# model should have already been loaded in bg_load_prepare_and_run. 
# If not, load it here    

WHICH_GLOVE = 'glove.6B.200d' #'glove.6B.100d' #'glove.840B.300d', 
#glove.twitter.27B.100d

glove_dir = '/Users/stefgarasto/Local-Data/wordvecs/'

LOADGLOVE = True
if LOADGLOVE:
    print('Loading glove model')
    t0 = time.time()
    # load the glove model
    model = gensim.models.KeyedVectors.load_word2vec_format\
    (os.path.join(glove_dir, 'word2vec.{}.txt'.format(WHICH_GLOVE)))
    #model = api.load("glove-wiki-gigaword-100")  # load pre-trained word-vectors
    # from gensim-data
    #model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
    #word_vectors = model.wv
    print_elapsed(t0, 'loading the glove model')
    
    vector_matrix = model.vectors
    list_of_terms = model.index2word

    lookup_terms = [convert_from_undersc(elem) for elem in list_of_terms]
    
# In[20]:
def get_keywords_list(x, stopwords):
    #all_keywords = []
    #x = df_line['Keywords']
    if isinstance(x, list):
        # I think that ik can be a collection of words separated by ";" or ","
        ik_elems0 = ' '.join([elem for elem in x if not elem.isdigit()])
        ik_elems0 = ik_elems0.replace('-', ' ').replace(':','').replace(',',';')
        ik_elems0 = ik_elems0.replace('(','').replace(')','')
        ik_elems0 = ik_elems0.split(';')
        # remove extra spaces and make lowercase
        ik_elems0 = [elem.strip().lower() for elem in ik_elems0]
        ik_elems0 = [elem for elem in ik_elems0 if len(elem)]
        ik_elems = []
        for ik_elem in ik_elems0:
            ik_elem0 = ' '.join([elem for elem in ik_elem.split() if
                  (not elem.isdigit()) & (elem not in stopwords) & len(elem)])
            if len(ik_elem0):
                ik_elems.append(ik_elem0)
                    #print(ik_elems)
        return [elem.strip() for elem in ik_elems if len(elem)]
    elif isinstance(x,str):
        #ik_elems = re.findall(r"[\w']+", df.loc[ix].replace('-',''))
        ik_elems = x.replace('-', ' ').replace(',',';')
        ik_elems = ik_elems.replace('(','').replace(')','').split(';')
        if len(ik_elems)==1:
            # lacking proper separators - have to use spaces
            ik_elems = ik_elems[0].split()
        # remove extra spaces
        ik_elems = [elem.strip().lower() for elem in ik_elems]
        # remove digits
        ik_elems = [elem for elem in ik_elems if not elem.isdigit()]
        ik_elems = [elem for elem in ik_elems if elem not in stopwords]
        ik_elems = [elem for elem in ik_elems if len(elem)]
        return [elem.strip() for elem in ik_elems if len(elem)]
    
    
#%%
def process_keywords(x, some_model):
    assert(isinstance(x, list))
    x_new= [' '.join(lemmatise_pruned(nltk.pos_tag(elem.split()))) for elem in x]
    x_new = [elem.strip() for elem in x_new if len(elem)]
    return x_new
            
#%%
def extract_tagged_title(df_line):
    title = df_line.loc['NOS Title']
    title_len = len(title.replace('-',' ').split())
    return df_line.loc['tagged_tokens'][:title_len]

#%%
def replace_skills_with_public(xs, public_skills, skills_matches):
    xs_new = []
    for x in xs.split('\n'):
        x = x.lower()
        if x in public_skills:
            if x.capitalize() not in xs_new:
                xs_new.append(x.capitalize())
        elif x in skills_matches:
            xs_new.append(skills_matches[x])
        else:
            out = process.extract(x, public_skills)
            out_values = [t[1] for t in out]
            if any([t>89 for t in out_values]):
                ix = np.argmax(out_values)
                y = out[ix][0]
                if y.capitalize() not in xs_new:
                    xs_new.append(y.capitalize())
    return '\n'.join(xs_new)


"""
Utils specific to mapping to requirements
"""

#%% load the dictionaries mapping SOC and SC to requirements if necessary
if not 'cols_v_clus' in locals():
    with open(os.path.join(saveoutput,'cols_v_clus3.pickle'), 'rb') as f:
        cols_v_clus = pickle.load(f)

#% SOC
if not 'cols_v_occ' in locals():
    with open(os.path.join(saveoutput,'cols_v_occ3.pickle'), 'rb') as f:
        cols_v_occ = pickle.load(f)
    
#% SOC+cluster
if not 'cols_v_occ_and_clus' in locals():
    with open(os.path.join(saveoutput,'cols_v_occ_and_clus3.pickle'), 'rb') as f:
        cols_v_occ_and_clus = pickle.load(f)



#%%
def map_nos_to_req(x,col,cols_v_clus):
    # this work the same whether the mapping is done via occupations or
    # skills cluster. All that matters is that x comes from the right column
    # to match one of the dictionary keys
    if isinstance(x, float):
        if not np.isnan(x):
            if x in matches_oobsoc_to_soc2:
                x = matches_oobsoc_to_soc2[x]
            return cols_v_clus[x][col + '-peak']
        else:
            return np.nan
    else:        
        if x in matches_oobsoc_to_soc2:
            x = matches_oobsoc_to_soc2[x]
        return cols_v_clus[x][col + '-peak']

#%%
def map_nos_to_req_dist(x,col,cols_v_clus):
    # this work the same whether the mapping is done via occupations or
    # skills cluster. All that matters is that x comes from the right column
    # to match one of the dictionary keys
    if isinstance(x, float):
        if not np.isnan(x):
            if x in matches_oobsoc_to_soc2:
                x = matches_oobsoc_to_soc2[x]
            return cols_v_clus[x][col]
        else:
            return np.nan
    else:        
        if x in matches_oobsoc_to_soc2:
            x = matches_oobsoc_to_soc2[x]
        return cols_v_clus[x][col]

#%%
#%
def map_nos_to_req2(x,col,cols_v_clus):
    # this function is also for a mapping by BOTH socs and clusters. 
    if not np.isnan(x):
        if x in matches_oobsoc_to_soc2:
            x = matches_oobsoc_to_soc2[x]
        return cols_v_clus[x][col + '-peak']
    else:
        return np.nan

#%
def map_nos_to_req_dist2(x,col,cols_v_clus):
    # this function is also for a mapping by BOTH socs and clusters. 
    # All that matters is that x comes from the right column
    # to match one of the dictionary keys
    if not np.isnan(x):
        if x in matches_oobsoc_to_soc2:
            x = matches_oobsoc_to_soc2[x]
        return cols_v_clus[x][col]
    else:
        return np.nan

#%%
def combine_requirements(x,col):
    ks = list(x.keys())
    if isinstance(x[ks[0]], Counter):
        Nk = len(ks)
        N = 0
        y = Counter()
        for k in ks:
            counter_values = list(x[k].values())
            counter_keys= list(x[k].keys())
            # normalise values
            N += np.sum(counter_values)
            counter_values = counter_values/np.sum(counter_values)/Nk
            y = y + Counter(dict(zip(counter_keys, counter_values)))
        # final de-normalisation (the values should still sum up to 1,
        # so multiplying by the sum of elements should give a stand-in for the
        # absolute values)
        counter_keys = list(y.keys())
        counter_values = [t*N for t in (y.values())]
        y = Counter(dict(zip(counter_keys, counter_values)))
        
    elif isinstance(x[ks[0]], float):
        y = np.mean(list(x.values()))
    elif isinstance(x[ks[0]], np.ndarray):
        # concatenate all salary arrays
        try:
            # if i can stack it's the skills embedding vectors
            y = np.stack(list(x.values()))
            y = np.mean(y, axis = 0)
        except:
            # if it fails, it's because salaries have different lengths
            y = np.concatenate(list(x.values()))
    elif isinstance(x[ks[0]], str):
        y = list(x.values())
    return y

#%
def map_nos_to_req_hub(x,col):
    y = {}
    for ix in x:
        if ix in cols_v_occ_and_clus:
            y[ix] = map_nos_to_req_dist(ix, col, cols_v_occ_and_clus)
        else:
            ix_split = ix.split('+')
            # try to match by SOC first
            s = float(ix_split[0])
            if s in cols_v_occ:
                y[s] = map_nos_to_req_dist(s, col, cols_v_occ)
            elif ix_split[1] in cols_v_clus:
                # if nothing else, map by skills cluster
                y[ix_split[1]] = map_nos_to_req_dist(ix_split[1], col, cols_v_clus)
            else:
                # if all fails, just do nothing
                tmp = None
    if len(y)==0:
        # what did I use for empty?
        return 'empty'
    # remove possible duplicates
    y_new = {}
    for iy in y:
        if not iy in y_new:
            y_new[iy] = y[iy]
    if len(y_new)>1:
        # combine the requirements
        return combine_requirements(y_new,col)
    else:
        k = list(y_new.keys())
        return y_new[k[0]]
        
    
#%
def join_soc_cluster(df_line):
    s = df_line['SOC4']
    if isinstance(s, float):
        if s in matches_oobsoc_to_soc2:
            s = matches_oobsoc_to_soc2[s]
    else:
        for ix,isoc in enumerate(s):
            if isoc in matches_oobsoc_to_soc2:
                s[ix] = matches_oobsoc_to_soc2[isoc]
    c = df_line['best_cluster_nos']
    if isinstance(s,float) & isinstance(c,str):
        return ['+'.join([str(s), c])]
    elif isinstance(s,list) & isinstance(c,str):
        y = []
        for isoc in s:
            y.append('+'.join([str(isoc),c]))
        return y
    elif isinstance(s,list) & isinstance(c,list):
        # both are lists
        y = []
        for isoc in s:
            for iclus in c:
                y.append('+'.join([str(isoc),iclus]))
        return y
    elif isinstance(s,float) & isinstance(c,list):
        y = []
        for iclus in c:
            y.append('+'.join([str(s),iclus]))
        return y 

#%  

def SC_to_requirements(df_nos_select, KEY= 'socs+clusters'):
    # ### Assign each job advert to a skill cluster    
    # how are we going to match them?
    #KEY = 'socs+clusters'
    #if not KEY == 'socs':
    #    assert(JA2CLUS)
    t0 = time.time()
    if KEY == 'socs':
        keycol = 'SOC4'
        keydict = cols_v_occ
    elif KEY=='clusters':
        keycol = 'best_cluster_nos'
        keydict = cols_v_clus
    elif KEY == 'socs+clusters':
        keycols = ['SOC4','best_cluster_nos']
        keycol = 'tmp'
        keydict = cols_v_occ_and_clus
    
    cols2match = ['myExp','myEdu','Salary','myExp-peak','myEdu-peak','Salary-peak',
                  'title_processed', 'converted_skills','London']
    
    
    '''
    # get the unmatched SOCs, that is the ones not in the OOB SOCs and not in the 
    # BG dataset
    '''
    
    socs_in_df = df_nos_select['SOC4'].map(lambda x: np.array(x)
                    if isinstance(x,(list,float)) else x).value_counts()
    socs_in_df = set(flatten_lol(list(socs_in_df.index.map(lambda x: [x] 
                    if isinstance(x,float) else list(x)))))
    unmatched_socs = set(socs_in_df) - set(list(cols_v_occ.keys()))
    unmatched_socs = list(unmatched_socs - set(matches_oobsoc_to_soc2.keys()))
    # select NOS with no SOC or with "unmacthed" SOC
    #nosoc = (df_nos_select['SOC4'].isnull()) | (df_nos_select['SOC4'].map(
    #        lambda x: x in unmatched_socs))
    
    # for the rows without SOC code, join by skill clusters
    if KEY == 'socs+clusters':
        #df_nos_select['tmp'] = df_nos_select[keycols[0]].map(lambda x: str(x) + '+'
        #        ) + df_nos_select[keycols[1]]
        df_nos_select['tmp'] = df_nos_select.apply(join_soc_cluster, axis = 1)
        
        for col in cols2match:
            if col not in ['myExp-peak','myEdu-peak']:
                df_nos_select[col] = df_nos_select['tmp'].map(lambda x:
                    map_nos_to_req_hub(x, col))
            else:
                # select the peak category again because some might have changed
                category = col.split('-')[0]
                df_nos_select[col] = df_nos_select[category].map(lambda x: 
                    x.most_common()[0][0] if isinstance(x,Counter) else None)
            
        '''
        # if a specific combo is not in the job advert data as well, 
        # just use the SOC code
        flag = df_nos_select['tmp'].map(lambda x: x not in cols_v_occ_and_clus)
        #flag = flag & (~nosoc)
        df_nos_select['tmp'][flag & (~nosoc)] = df_nos_select['SOC4'][flag & (~nosoc)]
        for col in cols2match:
            # first match by skill cluster for those that don't have a soc.
            # This is the ultimate fall-back if I have no other (better) strategy
            df_nos_select[col] = 'empty'        
            df_nos_select[col][nosoc] = df_nos_select['best_cluster_nos'][nosoc].map(
                                    lambda x: map_nos_to_req_dist(x,col,cols_v_clus))
            
            # now override by desired key if the key is present in the BG data
            df_nos_select[col][(~flag) & (~nosoc)] = df_nos_select[keycol
                           ][(~flag) & (~nosoc)].map(
                               lambda x: map_nos_to_req_dist(x,col,keydict))
            
            # now override by SOC for NOS with SOC but with a "bad" combo
            df_nos_select[col][(flag) & (~nosoc)] = df_nos_select[keycol
                           ][(flag) & (~nosoc)].map(
                               lambda x: map_nos_to_req_dist(x,col,cols_v_occ))
        '''
            
        del df_nos_select['tmp']
    else:
        for col in cols2match:
            df_nos_select[col] = df_nos_select[keycol].map(
                                    lambda x: map_nos_to_req_dist(x,col,keydict))
            
        
    print('Time to assign requirements: {:.4f}'.format(time.time()-t0))
    return df_nos_select[cols2match]


#%%
def match_super_suite_names(df_nos,super_suites_names,super_suites_files):

    all_super_suites = {}
    for which_super_suite in super_suites_names:
        all_super_suites[which_super_suite] = pd.read_excel(super_suites_files, 
                        sheet_name = which_super_suite)
        all_super_suites[which_super_suite]['NOS Suite name'] = all_super_suites[
            which_super_suite]['NOS Suite name'].map(
            lambda x: x.replace('(','').replace('(','').replace('&','and').strip().lower())
    
    # match the given suite names to the ones I extracted/got from ActiveIS
    standard_labels = list(df_nos.groupby('One_suite').groups.keys())
    all_matches = {}
    all_match_names = {}
    #match_name = []
    for which_super_suite in super_suites_names:
        all_matches[which_super_suite] = []
        for suite in all_super_suites[which_super_suite]['NOS Suite name'].values:
            # do manually some selected suites
            if 'insurance claims' in suite:
                tmp = standard_labels.index('general insurance')
                all_matches[which_super_suite].append(tmp)
                continue
            # for the "management and leadership marketing 2013" both marketing 
            # and marketing 2013 would fit,
            # but I'm only taking the latter
            # find a fuzzy match between 
            out = process.extract(suite, standard_labels, limit=3)
            if len(out) and out[0][1]>89:
                # note: most of them are above 96% similarity (only one is 90%)
                tmp = standard_labels.index(out[0][0])
                #print(suite, out[0])
                if tmp not in all_matches[which_super_suite]:
                    all_matches[which_super_suite].append(tmp)
                else:
                    if suite == 'installing domestic fascia, soffit, and bargeboards':
                        # this suite is kind of a duplicate - I aggregated it in my suites list
                        continue
                    tmp = standard_labels.index(out[2][0])
                    all_matches[which_super_suite].append(tmp)
                    print(out[0][0],',',out[1][0],',',out[2][0],',',suite)
            else:
                print(suite, ' not found')
                print(out)
                print('\n')
        print(len(all_matches[which_super_suite]),len(all_super_suites[which_super_suite]))
        all_match_names[which_super_suite] = [standard_labels[t] 
                        for t in all_matches[which_super_suite]]

    return all_super_suites, standard_labels, all_matches, all_match_names


#%%
    
def map_to_cluster(new_top_terms_dict,
                   new_top_weights_dict, comparison_vecs, clus_names,
                   USE_WEIGHTED= False):
    out2= []
    weights = 1
    # if no keywords, we can't make a decision
    if len(new_top_terms_dict)==0:
        out = ['uncertain']
        out2 = ['uncertain']
        return out, out2,[]
    test_skills, full_test_skills  = get_mean_vec(new_top_terms_dict, model, 
                                                  weights = new_top_weights_dict)
    counts_clus0 = []
    weighted_clus_dict = {}
    for iskill in range(full_test_skills.shape[0]):
        top_clusters = highest_similarity_threshold_top_two(full_test_skills[iskill], 
                                                   comparison_vecs, clus_names,
                                                   th = 0.4)
        #if new_top_terms_dict[iskill] not in feat_to_clusters.keys():
        #    feat_to_clusters[new_top_terms_dict[iskill]] = top_clusters
        # to sum the importances of each keywords
        for iclus in top_clusters:
            if iclus in weighted_clus_dict:
                weighted_clus_dict[iclus] += new_top_weights_dict[iskill]
            else:
                weighted_clus_dict[iclus] = new_top_weights_dict[iskill]
        # to just get the counts
        counts_clus0.append(top_clusters)
    counts_clus = flatten_lol(counts_clus0)
    if len(counts_clus)==0:
        # no good keywords here, anywhere
        out = ['uncertain']
        out2 = ['uncertain']
        return out, [], out2, []
    # backup the skills cluster associated with the most important feature
    tmp_bkp= counts_clus[0]
    counts_clus = Counter(counts_clus).most_common()
    weighted_clus = Counter(weighted_clus_dict).most_common()
    #full_clusters_counter = [counts_clus, weighted_clus]
    
    # assign to a skills cluster
    # first case: only one skills clusters have been identified
    if len(counts_clus)==1:
        # easy
        out = counts_clus[0][0]
    # second case: there are just two skills clusters
    elif len(counts_clus)==2:
        # they occur an equal number of times
        if counts_clus[0][1]==counts_clus[1][1]:
            # each has been identified only once: take them both
            if counts_clus[0][1]==1:
                out = [t[0] for t in counts_clus]
            # each occur more than once
            elif counts_clus[0][1]>1:
                # can I break the tie using the tfidf weights?
                if USE_WEIGHTED:
                    out = weighted_clus[0][0]
                else:
                    # take them both
                    out = [t[0] for t in counts_clus]
        # they are not the same
        else:
             #take the first one
             out = counts_clus[0][0]
    # third case: multiple skills clusters identified, but just once each
    elif all([t[1]==1 for t in counts_clus]):
        out = highest_similarity_threshold_top_two(test_skills, comparison_vecs, 
                 clus_names, th = 0.3) #tmp_bkp
    
    # other cases: more than one cluster, at least one identified multiple times
    else:
        # should I use the weights?
        if USE_WEIGHTED:
            out = weighted_clus[0][0]
        # are the first two the same?
        elif counts_clus[0][1]==counts_clus[1][1]:
            # take them both
            out = [t[0] for t in counts_clus[:2]]
        else:
            # take the first one
            out = counts_clus[0][0]
    if isinstance(out,str):
        out = [out]
    return out, out2, counts_clus, weighted_clus, counts_clus0


#%%
def assign_third_cluster(row, level = 'third'):
    '''
    '''
    if isinstance(row,pd.core.series.Series):
        if level == 'third':
            counts_clus_third = row['exact_third_level']
            counts_clus_second = row['exact_second_level']
        else:
            counts_clus_third = row['exact_second_level']
            counts_clus_second = row['exact_first_level']
    elif isinstance(row,tuple):
        counts_clus_third, counts_clus_second = row
    # assign to a skills cluster
    counts_clus = Counter(counts_clus_third).most_common()
    counts_clus2 = Counter(counts_clus_second)
    # first case: only one skills clusters have been identified
    if len(counts_clus)==1:
        # easy
        out_cluster = counts_clus[0][0]
    # second case: there are just two skills clusters
    elif len(counts_clus)==2:
        # they occur an equal number of times
        if counts_clus[0][1]==counts_clus[1][1]:
            # take them both
            out_cluster = [t[0] for t in counts_clus]
        # they are not the same
        else:
             #take the first one
             out_cluster = counts_clus[0][0]
    # third case: multiple skills clusters identified, but just once each
    elif all([t[1]==1 for t in counts_clus]):
        # if also the second level clusters are all different, then there's too
        # much uncertainty
        if all([counts_clus2[t]==1 for t in counts_clus2]):
            out_cluster= ['uncertain']
        else:
            # if not can I break the tie using second level clusters?
            # build a dataframe for this
            clusters_df= pd.DataFrame(zip(counts_clus_third,counts_clus_second,counts_clus_third))
            # get most frequent second level clusters
            best_second_level = clusters_df[1].value_counts()
            # only keep those that appear more often
            best_second_level = best_second_level[best_second_level==best_second_level[0]]
            if len(best_second_level)==1:
                # if just one remains pick at most two third level clusters from there
                best_third_clusters = clusters_df[clusters_df[1]==best_second_level.index[0]][0]
                best_third_clusters = best_third_clusters.value_counts()
                if len(best_third_clusters)<=2:
                    # if there are two different ones at most, then great
                    out_cluster= list(best_third_clusters.index)
                else:
                    # just take two random ones
                    out_cluster= list(best_third_clusters.sample(n=2).index)
            elif len(best_second_level)==2:
                # take one random third level cluster from each good second level cluster
                # very shady I know
                out_cluster = []
                #clusters_df_groups = clusters_df.groupby(1)
                for name in best_second_level.index:
                    best_third_clusters = clusters_df[clusters_df[1]==name]
                    out_cluster.append(best_third_clusters.sample(n=1)[0].iloc[0])
            else:
                # still too uncertain
                out_cluster= ['uncertain']
    # other cases: more than one cluster, at least one identified multiple times
    else:
        # are the first two the same?
        if counts_clus[0][1]==counts_clus[1][1]:
            # take them both
            out_cluster = [t[0] for t in counts_clus[:2]]
        else:
            # take the first one. Note, this might also mean that the first 3 ones
            # are the same and I'm only taking the first one, but oh well
            out_cluster = counts_clus[0][0]
    if isinstance(out_cluster,str):
        out_cluster = [out_cluster]
    return out_cluster










