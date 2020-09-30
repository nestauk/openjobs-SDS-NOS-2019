#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:45:41 2019

@author: stefgarasto
"""
#checkpoint
FIRST_RUN = True
if FIRST_RUN:
    print('Setting up all the necessary structures')
    #get_ipython().run_line_magic('matplotlib', 'inline')
    import matplotlib.pyplot as plt
    
    
    import os
    #import itertools
    #import json
    import numpy as np
    import pandas as pd
    import pickle
    #import requests
    import seaborn as sns
    #import collections
    from collections import Counter
    import scipy
    from scipy.cluster.vq import whiten, kmeans
    import time
    #import copy
    from collections import OrderedDict, Counter
    
    import matplotlib as mpl
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Rectangle
    
    
    #from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    #from sklearn.decomposition.pca import PCA
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn import mixture
    
    import nltk
    #from nltk.stem import PorterStemmer
    #from nltk.stem.snowball import SnowballStemmer
    
    import gensim
    import re
    from fuzzywuzzy import process
    from ast import literal_eval
    #from gensim.scripts.glove2word2vec import glove2word2vec
    
    
    # In[163]:
    
    
    from utils_bg import * #nesta_colours, nesta_colours_combos
    from utils_nlp import *
    from utils_skills_clusters import load_and_process_clusters, skills_matches
    from utils_skills_clusters import public_skills, public_skills_full
     
    from map_NOS_to_pathways_utils import *
    
# In[225]:
SAVEFIG = True

supersuite = 'engineering'
empty = Counter()
# checkpoint
LOAD_DATA = True
if LOAD_DATA:
    print('Loading the data')
    df_nos_select = pd.read_csv(output_dir + 
                         '/augmented_info_NOS_in_supersuites_{}.csv'.format(supersuite))
    df_nos_select.set_index('Unnamed: 0', inplace = True)
    # need to evaluate some columns
    for col in ['SOC4','SOC3','SOC2','SOC1','pruned_lemmas']:
        print('Processing',col)
        df_nos_select[col]= df_nos_select[col].map(literal_eval)
    for col in ['Salary-peak']:
        print('Processing',col)
        flag = df_nos_select[col].map(lambda x: isinstance(x,str) & (x!='empty'))
        df_nos_select[col][flag] = df_nos_select[col][flag].map(literal_eval)
    for col in ['best_cluster_nos']:
        print('Processing',col)
        flag = df_nos_select[col].map(lambda x: (x[0]=='[') & (x!='empty'))
        df_nos_select[col][flag] = df_nos_select[col][flag].map(literal_eval)
    for col in ['myExp','myEdu','title_processed','converted_skills','London']:
        print('Processing',col)
        df_nos_select[col]= df_nos_select[col].map(lambda x: eval(x))
    for col in ['Salary']:
        print('Processing',col)
        df_nos_select[col] =df_nos_select[col].map(lambda x: re.findall(r'[0-9]+',x)).map(
                lambda x: [float(t) for t in x])
    
    #%%
    '''
    qualifier = 'postjoining_final_no_dropped'
    qualifier0 = 'postjoining_final_no_dropped'
    paramsn = {}
    pofs = 'nv'
    pofs_titles = 'nvj'
    
    paramsn = {}
    paramsn['ngrams'] = 'bi'
    paramsn['pofs'] = pofs
    USE_TITLES = True
    USE_KEYWORDS = False
    assert(USE_TITLES)
    assert(not USE_KEYWORDS)
    if USE_KEYWORDS:
        paramsn['tfidf_min'] = 2
        paramsn['tfidf_max'] = 0.9
    elif USE_TITLES:
        paramsn['tfidf_min'] = 2
        paramsn['tfidf_max'] = 0.4
    else:
        paramsn['tfidf_min'] = 3
        paramsn['tfidf_max'] = 0.4
    
    
    paramsn['bywhich'] = 'docs'
    paramsn['mode'] = 'tfidf'

    # Load stopwords
    with open(lookup_dir + 'stopwords_for_nos_{}_{}.pickle'.format(qualifier,pofs),'rb') as f:
        stopwords0, no_idea_why_here_stopwords, more_stopwords = pickle.load(f)
    stopwords = stopwords0 + no_idea_why_here_stopwords 
    stopwords += tuple(['¤', '¨', 'μ', 'บ', 'ย', 'ᶟ', '‰', '©', 'ƒ', '°', '„'])
    stopwords0 += tuple(['¤', '¨', 'μ', 'บ', 'ย', 'ᶟ', '‰', '©', 'ƒ', '°', '„',
                     "'m", "'re", '£','—','‚°','●'])
    #stopwords0 += tuple(set(list(df_nos_select['Developed By'])))
    stopwords0 += tuple(['cosvr','unit','standard','sfl','paramount','tp','il',
                         'al','ad','hoc','someone','task','gnem','role',
                         'fin','flake','multiple','greet','go','find',
                         'agreed','agree','give','un','day','lloyd',
                         'whatever','whoever','whole','try','week','year','years',
                         'say','quo','1', '@','legacy','&','\uf0b7'])
    stopwords0 += tuple(['system','systems'])   
    
    tfidf_n = define_tfidf(paramsn, stopwords0)
    
    # get the transform from the whole NOS corpus
    FULL_CORPUS = False
    if FULL_CORPUS:
        _, feature_names_n, tfidf_n, _ = get_tfidf_matrix(
            paramsn, df_nos_select, tfidf_n, col = 'pruned_lemmas')
    else:
        #df_nos_select = df_nos[df_nos['supersuite']=='engineering']
        _, feature_names_n, tfidf_n, _ = get_tfidf_matrix(
            paramsn, df_nos_select, tfidf_n, col = 'pruned_lemmas')
    '''
    # load TFIDF transform
    tfidf_file = output_dir + '/augmented_tfidf_NOS_in_supersuites_{}.pickle'.format(supersuite)
    with open(tfidf_file,'rb') as f:
        feature_names_n, tfidf_n = pickle.load(f)
        
stop
#%%
# If I want to save all the indicators
#Range of values from the first quartile (25% percentile) to the third quartile (75% percentile) of the salary distribution
# Interquartile range of the offered salary distribution
SAVE_IND = True
if SAVE_IND:
    df_nos = df_nos_select[['NOS Title', 'supersuite', 'One_suite', 'SOC4','URN', 
                            'best_cluster_nos','myExp', 'myEdu', 'Salary', 'myExp-peak',
                            'myEdu-peak', 'Salary-peak']]
    df_nos = df_nos.rename(columns = {'NOS Title':'NOS title', 'One_suite': 'Suite', 
                                      'SOC4': 'SOC code','URN':'URN', 
                                      'best_cluster_nos':'Best-fit skill cluster',
                                      'myExp':'Experience requirements', 
                                      'myEdu':'Education requirements', 
                                      'Salary':'Offered salary',
                                      'myExp-peak': 'Top experience requirement', 
                                      'myEdu-peak':'Top education requirement', 
                                      'Salary-peak':'Average salary'})
    
    #df_nos['N'] = df_nos['Education requirements'].map(lambda x: x['Pregraduate']+x['Graduate']+x['Postgraduate'])
    
    def get_percentage_exp(x, category):
        N=x['Entry-level']+x['Mid-level']+x['Senior-level']
        if N==0:
            return 0.0
        else:
            return np.around(x[category]/N,3)*100
            
    def get_percentage_edu(x, category):
        N=x['Pregraduate']+x['Graduate']+x['Postgraduate']
        if N==0:
            return 0.0
        else:
            return np.around(x[category]/N,3)*100
        
    def take_prc_local(x,p):
        try:
            return np.percentile(x,p)
        except:
            return 0.0
    
    def adjust_skill_cluster(x):
        if isinstance(x,str):
            if x == 'uncertain':
                return 'NA'
            else:
                return x.capitalize()
        elif isinstance(x,list):
            return '; '.join([t.capitalize() for t in x])
        else:
            return 'NA'
        
    df_nos['Pregraduate percentage'] = df_nos['Education requirements'].map(
            lambda x: get_percentage_edu(x,'Pregraduate'))
    df_nos['Graduate percentage'] = df_nos['Education requirements'].map(
            lambda x: get_percentage_edu(x,'Graduate'))
    df_nos['Postgraduate percentage'] = df_nos['Education requirements'].map(
            lambda x: get_percentage_edu(x,'Postgraduate'))
    df_nos['Senior level percentage'] = df_nos['Experience requirements'].map(
            lambda x: get_percentage_exp(x,'Senior-level'))
    df_nos['Mid level percentage'] = df_nos['Experience requirements'].map(
            lambda x: get_percentage_exp(x,'Mid-level'))
    df_nos['Entry level percentage'] = df_nos['Experience requirements'].map(
            lambda x: get_percentage_exp(x,'Entry-level'))
    df_nos['Best-fit skill cluster'] = df_nos['Best-fit skill cluster'].map(
            adjust_skill_cluster)
    
    df_nos['Median salary'] = df_nos['Offered salary'].map(np.median)
    df_nos['Salary first quartile'] = df_nos['Offered salary'].map(lambda x: 
        take_prc_local(x,25))
    df_nos['Salary third quartile'] = df_nos['Offered salary'].map(lambda x: 
        take_prc_local(x,75))
    
    df_nos2 = df_nos[['NOS title', 'Suite', 'SOC code', 'URN', 
                      'Best-fit skill cluster', 'Top experience requirement', 
                      'Top education requirement', 'Average salary', 
                      'Salary first quartile', 'Median salary', 'Salary third quartile', 
                      'Pregraduate percentage', 'Graduate percentage', 
                      'Postgraduate percentage', 'Senior level percentage',
                      'Mid level percentage', 'Entry level percentage']]
    
            
    df_nos2['Average salary'][df_nos2['Average salary'].map(lambda x: isinstance(x,str))]=0
    df_nos2['Average salary'] = df_nos2['Average salary'].map(lambda x: np.around(x,2))
    for col in ['Average salary', 'Salary first quartile','Salary third quartile']:
        df_nos2[col][df_nos2[col].map(lambda x: x==0)]= 'NA'
    df_nos2['Median salary'][df_nos2['Median salary'].map(lambda x: np.isnan(x))] = 'NA'
    df_nos2['Average salary'][df_nos2['Median salary'].map(lambda x: x=='NA')] = 'NA'
    
    
    df_nos2['Top experience requirement'][df_nos2['Top experience requirement'].map(
            lambda x: isinstance(x,float))]='NA'
    df_nos2['Top education requirement'][df_nos2['Top education requirement'].map(
            lambda x: isinstance(x,float))]='NA'
    df_nos2['Top experience requirement'] = df_nos2['Top experience requirement'].map(
            lambda x: x.replace('-',' '))
    
    df_nos2.to_csv(
     '/Users/stefgarasto/Google Drive/Documents/Outputs/NOS_summary/WP2_files2/Progression_indicators_for_engineering_NOS.csv')

#%% checkpoint
LOAD_CLUSTERS= True
'''
Here is where I take a sub selection of NOS / suites from all the possible
NOS in the super-suites.

'''
if LOAD_CLUSTERS:
    #%%
    print('Loading the clusters')
    STRATEGY = 'hdbscan_tfidfs_joined' #'tfidf' #'we'
    #pofs_clusters = 'n' #'nv' #'n'
    
    clusters2use_f = ''.join(['/Users/stefgarasto/Google Drive/Documents/results/',
                              'NOS/nlp_analysis/svq_qualifications_n_postjoining_',
                              'final_no_dropped/all_nos_with_svq_clusters_select_',
                              'in_engineering_tfidf_n.xlsx'])
    
    sheet = STRATEGY.replace('_','') #'hierarchicaltfidfa' #'hierarchical' #'hierarchicalward'
    #%
    clusters2use = pd.read_excel(clusters2use_f, sheet_name= sheet).T
    
    clusters2use = clusters2use[~clusters2use[0].isnull()]
    
    rename_dict = {}
    
    for ix in range(4):
        if isinstance(clusters2use[ix].iloc[0],int):
            start_of_titles = ix
            break
        rename_dict[ix] = clusters2use[ix].iloc[0]
    clusters2use = clusters2use.rename(columns = rename_dict)
    clusters2use = clusters2use[1:]
    clusters2use['Cluster index'] = clusters2use['Cluster index'].map(literal_eval)
    #clusters2use = clusters2use.set_index(rename_dict[0])
    
    #%%
    labels_folder = '/Users/stefgarasto/Google Drive/Documents/results/NOS/nlp_analysis/svq_qualifications_n_postjoining_final_no_dropped/'
    if 'hdbscan' in sheet:
        labels_f = os.path.join(labels_folder, 
                    ''.join(['hdbscan_nos_cut_labels_in_engineering_svq_postjoining',
                             '_final_no_dropped_uni_tfidf_substitute.csv']))
        nos_clusters = pd.read_csv(labels_f)
        nos_clusters = nos_clusters.rename(columns = {'hierarchical': 'labels'})
    
    elif 'hierarchicaltfidf' in sheet:
        labels_f = os.path.join(labels_folder, ''.join(['','']))
        nos_clusters = pd.read_csv(labels_f)
        nos_clusters = nos_clusters.rename(columns = {'hierarchical': 'labels'})
        
    elif 'cliques' in sheet:
        labels_f = os.path.join(labels_folder, ''.join(['','']))
        nos_clusters = pd.read_csv(labels_f)
        nos_clusters = nos_clusters.rename(columns = {'hierarchical': 'labels'})
        
    else:
        raise ValueError

    
    #%%
    # first add new columns we're going to use
    df_nos_select['N'] = df_nos_select['myEdu'].map(lambda x: x['Pregraduate'] + x['Graduate']
                + x['Postgraduate'])
    for col in all_edu:
        df_nos_select[col] = df_nos_select['myEdu'].map(lambda x: x[col])
        df_nos_select[col] = df_nos_select[col]/df_nos_select['N']
    df_nos_select['centEdu']= df_nos_select[all_edu[0]]*1 + df_nos_select[all_edu[1]]*2 + df_nos_select[all_edu[2]]*3
    for col in all_exp:
        df_nos_select[col] = df_nos_select['myExp'].map(lambda x: x[col])
        df_nos_select[col] = df_nos_select[col]/df_nos_select['N']
    df_nos_select['centExp']= df_nos_select[all_exp[0]]*1 + df_nos_select[all_exp[1]]*2 + df_nos_select[all_exp[2]]*3
    
    #%%
    # eliminate any NOS with no requirements
    df_nos_select2 = df_nos_select[df_nos_select['N']>0]
    
    #%%
    # do PCA, jsut in case it's useful for later
    allvals = extract_vals(df_nos_select2,all_edu,all_exp)
    #allvals = np.array(df_nos_select[all_edu[:2] + all_exp[:2] + ['Salary-peak']].values, dtype =np.float32)
    #allvals[np.isnan(allvals[:,6]),6] = np.nanmean(allvals[:,6])
    x = StandardScaler().fit_transform(allvals)
    #x = whiten(allvals)
    pca = PCA(x.shape[1])#'mle')
    prinComp = pca.fit_transform(x)
    pca_cumsum = np.cumsum(pca.explained_variance_ratio_)
    Npca= np.where(pca_cumsum>.99)[0][0]+1
    #plt.figure(figsize = (8,4)),plt.plot(np.cumsum(pca.explained_variance_ratio_))
    print('Number of PCA components: ', Npca)
    # redo PCA with lower nb of components
    pca = PCA(Npca)
    prinComp = pca.fit_transform(x)
    #plt.figure(figsize = (8,4)),plt.plot(np.cumsum(pca.explained_variance_ratio_))

#%%
def aggregate_best_clusters(x):
    y = []
    for t in x:
        if isinstance(t,list):
            y += t
        else:
            y += [t]
    return Counter(y)

#%%
# checkpoint
DO_CLUS = True
SAVE_PATH = True
SAVE_SKILLS = True
KEY = 'socs+clusters'
#checkparams : this are related to how I process the titles to get 
skills_clusters_parameters = 'titles_nvj_bi_2_0.4'

if DO_CLUS:
    #%
    A = df_nos_select2['Salary'].map(lambda x: take_prc(x,98))
    xlimU = max(A)
    xlimL = 0
    output_dir2 = output_dir + '/nosclusters/{}_{}'.format(
                STRATEGY,skills_clusters_parameters)        
    USE_PCA = True
    if USE_PCA:
        output_dir2 += '_pca'
    for MODE in clusters2use.index: # cycle through the clusters
        
        SELECT_MODE = clusters2use.loc[MODE]
        
        ''' Here, do the plots for when I take the NOS from the clusters'''

        #% select the cluster (change the function - probably better to have a specific one for this case of mixed lists and numbers)
        final_nos, final_groups, larger_suites, cluster_name, cluster_name_save, \
            cluster_name_figs = select_subdf_from_list(SELECT_MODE, clusters2use, 
                                             nos_clusters,df_nos_select2)
        #final_nos['NOS Title'].to_csv(''.join(['/Users/stefgarasto/Google Drive/',
        #         'Documents/results/NOS/nlp_analysis/svq_qualifications_n_post',
        #         'joining_final_no_dropped/nos_titles_{}.csv'.format(cluster_name)]),
        #                            header = True)
    
        #% replace oob soc codes
        final_nos['SOC4'] = final_nos['SOC4'].map(replace_oob_socs)
        # remove NOS with legacy in the title
        #print('nb with legacy nos:',len(final_nos))
        final_nos = final_nos[final_nos['NOS Title'].map(lambda x: 'legacy' not in x)]
        final_nos = final_nos[final_nos.index.map(lambda x: not x[-5:]=='l.pdf')]
        #print('nb without legacy nos:',len(final_nos))
        final_nos['Salary-peak'] = final_nos['Salary-peak'].map(lambda x: np.float32(x))
        # if too few NOS are left, continue
        #if len(final_nos)<10:
        #    continue
        print('Nb of NOS to level for thematic group {} is: {}'.format(
                cluster_name, len(final_nos)))        
        #% extract skills that are important for each of the exp/edu pairs
        skillsdf =extract_top_skills(final_nos,all_exp,all_edu)
        if SAVE_SKILLS:
            skillsdf.to_csv(output_dir2 + '/NOS_topskills_for_{}_{}_v0.csv'.format(
                    cluster_name_save,KEY))

        #%
        KM = True
        print('Starting the clustering with GMM')
        KM_MODE = 'gmm'
        KMfunc = {'km': do_kmean, 'gmm': do_gmm, 'bgmm': do_bgmm}
        nos_levels = []
        if KM:
            #%
            vals = extract_vals(final_nos, all_edu, all_exp)
            xsmall = StandardScaler().fit_transform(vals)
            if USE_PCA:
                xsmall = pca.transform(xsmall)
            max_K_clus = np.unique(xsmall, axis =0).shape[0]
            if max_K_clus == 1:
                labels, clusterer, kmax, stab = KMfunc[KM_MODE](xsmall, 
                            ks = np.arange(1,2),N=10)
            else:
                labels, clusterer, kmax, stab = KMfunc[KM_MODE](xsmall, 
                            ks = np.arange(2,min([6,max_K_clus+1])),N=100)
            #%
            nos_levels.append((labels,clusterer,kmax,stab))
                
            #% now save the relevant info
            exp2num = {'Entry-level':1, 'Mid-level': 2,'Senior-level':3}
            edu2num = {'Pregraduate':1, 'Graduate': 2,'Postgraduate':3}
            rename_dict = {'myEdu-peak': 'Educational requirement',
                           'myExp-peak':'Experience requirement',
                           'Salary-peak': 'Avg salary',
                           'NOS Title': 'NOS titles', 'URN': 'URN',
                           'converted_skills': 'Top skills',
                           'skills_is_public': 'skills_is_public',
                           'skills_is_public_full': 'skills_is_public_full',
                           'title_processed': 'Top job titles',
                           'SOC4': 'Occupation','best_cluster_nos': 'Skills cluster',
                           'Top keywords': 'Top keywords',
                           'myExp': 'myExp',
                           'myEdu': 'myEdu'}
            # Collect info about each level
            final_nos['levels'] = nos_levels[0][0]
            group_levels = final_nos.groupby('levels')
            # edu/exp
            nos_groups = group_levels[['myExp-peak','myEdu-peak']]
            nos_groups = nos_groups.agg(lambda x: Counter(x).most_common()[0][0]) #np.max)
            # average salary
            nos_groups2 = group_levels['Salary-peak']
            nos_groups2 = nos_groups2.agg(np.median).map(np.round)
            nos_groups = nos_groups.join(nos_groups2)
            # NOS titles
            final_nos['NOS Title'] = final_nos['NOS Title'].map(
                    lambda x: x.capitalize())
            nos_groups2 = group_levels['NOS Title']
            nos_groups2 = nos_groups2.apply('\n'.join)
            nos_groups = nos_groups.join(nos_groups2)
            # URNs
            nos_groups2 = group_levels['URN']
            nos_groups2 = nos_groups2.apply('\n'.join)
            nos_groups = nos_groups.join(nos_groups2)
            # top 10 skills
            nos_groups2 = pd.DataFrame(group_levels['converted_skills'].agg(
                    'sum').map(lambda x:x.most_common()).map(lambda x: x[:20]).map(
                            lambda x: [t[0].capitalize() for t in x]).map('\n'.join))
            # mark skills that are not "public"
            #tmp = pd.DataFrame(group_levels['converted_skills'].agg(
            #        'sum').map(lambda x:x.most_common()).map(lambda x: x[:20]).map(
            #                lambda x: [t[0].capitalize() for t in x 
            #                           if t[0] in public_skills]).map('\n'.join))
            tmp = nos_groups2['converted_skills'].map(lambda x: 
                    replace_skills_with_public(x, public_skills))
            nos_groups2['skills_is_public'] = tmp#['converted_skills']
            #tmp = pd.DataFrame(group_levels['converted_skills'].agg(
            #        'sum').map(lambda x:x.most_common()).map(lambda x: x[:20]).map(
            #                lambda x: [t[0].capitalize() for t in x 
            #                           if t[0] in public_skills_full]).map('\n'.join))
            tmp = nos_groups2['converted_skills'].map(lambda x: 
                    replace_skills_with_public(x, public_skills_full))
            nos_groups2['skills_is_public_full'] = tmp#['converted_skills']
            nos_groups = nos_groups.join(nos_groups2)
            # top 10 job titles
            nos_groups2 = group_levels['title_processed'].agg(
                    'sum').map(lambda x:x.most_common()).map(lambda x: x[:10]).map(
                            lambda x: [t[0].capitalize() for t in x]).map('\n'.join)
            nos_groups = nos_groups.join(nos_groups2)
            # most common occupation
            nos_groups2 = group_levels['SOC4']
            nos_groups2 = nos_groups2.agg(aggregate_best_clusters).map(lambda x: x.most_common()
                ).map(lambda x: x[:3]).map(lambda x: [socnames_dict[t[0]].capitalize() 
                for t in x])
            #nos_groups2.agg(np.max).map(lambda x: socnames_dict[x])
            nos_groups = nos_groups.join(nos_groups2)
            # top 10 keywords
            # concatenate tokens 
            # (careful --> these might just be titles or expert keywords)
            #TODO: change this to always get data-driven keywords?
            tokens_concat = group_levels['pruned_lemmas'].agg(sum)
            # take transform
            tfidfm_tmp = tfidf_n.transform(pd.DataFrame(tokens_concat)[
                    'pruned_lemmas']).todense()
            nos_groups2 = {}
            ix=0
            for name, group in group_levels:
                top_ngrams, top_weights, top_features = extract_top_features(
                        tfidfm_tmp[ix,:], feature_names_n, N=10)
                M = np.min([10,len(top_features)])
                tmp = '\n '.join(['({}, {:.3f})'.format(top_features[ix], 
                                  top_weights[ix]) for ix in range(M)])
                nos_groups2[name] = tmp
                ix +=1
            nos_groups= nos_groups.join(pd.DataFrame.from_dict(nos_groups2, orient = 'index',
                                                     columns = ['Top keywords']))
            # most common skills cluster (?)
            nos_groups2 = group_levels['best_cluster_nos']
            nos_groups2 = nos_groups2.agg(aggregate_best_clusters).map(lambda x: x.most_common()
                ).map(lambda x: x[:3]).map(lambda x: [t[0].capitalize() 
                for t in x])
            #nos_groups2 = nos_groups2.agg(np.max)
            nos_groups = nos_groups.join(nos_groups2)
            # add full dist of exp, edu and salaries
            nos_groups2 = group_levels['myExp']#,'myEdu-peak']]
            nos_groups2 = nos_groups2.agg('sum')
            nos_groups = nos_groups.join(nos_groups2)
            nos_groups2 = group_levels['myEdu']#,'myEdu-peak']]
            nos_groups2 = nos_groups2.agg('sum')
            nos_groups = nos_groups.join(nos_groups2)
            # now sort them by edu then exp then salary
            nos_groups['myExp-num'] = nos_groups['myExp-peak'].map(lambda x: exp2num[x])
            nos_groups['myEdu-num'] = nos_groups['myEdu-peak'].map(lambda x: edu2num[x])
            nos_groups = nos_groups.sort_values(by=['myEdu-num','myExp-num','Salary-peak'])
            nos_groups = nos_groups[list(rename_dict.keys())]
            nos_groups = nos_groups.rename(columns = rename_dict)
            # save these potential pathways
            if SAVE_PATH:
                nos_groups.to_csv(output_dir2 + '/NOS_levels_for_{}_{}_v0_public.csv'.format(
                    cluster_name_save,KEY))
            
            
            
        #% for each cluster plot all NOS in order of increasing salary
        SAL = True
        if SAL:
            #for ix, nos_clus_df in enumerate(nos_clus_dfs):
            tmp_nos = final_nos.sort_values(by='Salary-peak')[['NOS Title','Salary']]
            # only take first title if it's a duplicate
            tmp_nos['NOS Title'] = tmp_nos['NOS Title'].map(lambda x: x.split(';')[0])
            tmp_nos2 = [tmp_nos['Salary'].loc[t] for t in tmp_nos.index]
            #xlimU = max([np.percentile(t,98) for t in tmp_nos2])
            #xlimL = 5000 #min([min(t) for t in A])
            fig = plt.figure(figsize = (12,.6*len(tmp_nos)))
            plt.boxplot(tmp_nos2, vert= False, notch = True, 
                        medianprops = {'color':nesta_colours[3]})
            plt.xlim([xlimL,xlimU])
            plt.yticks(np.arange(1,len(tmp_nos)+1), tmp_nos['NOS Title'].values)
            plt.tight_layout()
            plt.xticks(rotation = 30, ha="right",
                         rotation_mode="anchor")
            if SAVEFIG:
                plt.savefig(output_dir2 + 
                    '/NOS_ordered_by_salary_for_{}_{}_v0.png'.format(
                    cluster_name_save, KEY), bbox_inches='tight')
                plt.close(fig)
        
        #%
        # plot heatmap of skills clusters vs occupations
        SCVSSOC = False
        if SCVSSOC:
            if final_nos['SOC4'].isnull().sum()/len(final_nos)<0.9:
                # first remove any clusters that is a list of one
                final_nos['best_cluster_nos']= final_nos['best_cluster_nos'].map(
                        lambda x: x[0] if len(x)==1 else x)
                # TODO: need to divide by both SOCs and clusters
                divide_by_cluster = final_nos['best_cluster_nos'].map(
                        lambda x: isinstance(x,str))
                single_clus_nos = final_nos[divide_by_cluster]
                mult_clus_nos = final_nos[~divide_by_cluster]
                x_quantity = single_clus_nos['SOC4'].map(lambda x : socnames_dict[x])
                y_quantity = single_clus_nos['best_cluster_nos']
                for t in range(2):
                    x_quantity = x_quantity.append(mult_clus_nos['SOC4'].map(
                            lambda x : socnames_dict[x]))
                    y_quantity = y_quantity.append(
                            mult_clus_nos['best_cluster_nos'].map(lambda x:x[t]))
                hm2plot = pd.crosstab(x_quantity, y_quantity).T
                hm2plot = hm2plot/hm2plot.sum()
                h1, w1 = hm2plot.shape
                plotHM(hm2plot, normalize = False, w = .5*w1+6, h = .5*h1+2,
                           title = 'Occupations vs skills clusters (\'{}\')'.format(
                                                   cluster_name_figs))
                plt.tight_layout()
            else:
                # or just histogram of skills clusters
                tmp = final_nos['best_cluster_nos'].value_counts()
                fig = plt.figure(figsize = (len(tmp),7))
                tmp.plot('bar', color= nesta_colours[3])
                for ix,i in enumerate(tmp):
                    if i>3:
                        plt.text(ix-.2, i+.5,'{}'.format(i), fontsize = 14)
                plt.setp(plt.gca().get_xticklabels(), rotation=60, ha="right",
                             rotation_mode="anchor")
                plt.ylabel('NOS counts', fontsize = 18)
                plt.xlabel('Skills cluster', fontsize = 18)
                plt.title('Distribution of skills cluster across NOS (\'{}\')'.format(
                            cluster_name_figs), fontsize = 18)
                plt.tight_layout()
            if SAVEFIG:
                plt.savefig(output_dir2 +
                        '/NOS_skills_clusters_for_{}_{}_v0.png'.format(
                        cluster_name_save,KEY))
                plt.close(fig)
                
                
                
#%%
# ad-hoc requests for diagnosis
ADHOC = False
if ADHOC:
    # select NOS of interests
    A = df_nos_select2[df_nos_select2['NOS Title'].map(lambda x: 'contribute to technical leadership' in x)][
            ['NOS Title' ,'myExp','myEdu','best_cluster_nos','SOC4',
             'myExp-peak','myEdu-peak']]
    # print quantities
    for a in A.T:
        print(A.loc[a]['SOC4'])
    for a in A.T:
        print(A.loc[a]['best_cluster_nos'])
    for a in A.T:
        print(A.loc[a]['NOS Title'])
    for a in A.T:
        B = A.loc[a]['myEdu']
        print([(t,B[t]) for t in B])
    for a in A.T:
        B = A.loc[a]['myExp']
        print([(t,B[t]) for t in B])