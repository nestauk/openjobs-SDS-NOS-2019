#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:45:16 2019

@author: stefgarasto
"""

# coding: utf-8
#!/usr/bin/env python

'''
IMPORTANT NOTES

Here, I operate on groups of NOS, that is groups of duplicates identified by
the LSH algorithm.

I load the groups from the python script where I also cluster them (after removing
transferable NOS as well). However, all I can save about these groups are the joined
titles (and perhaps the list of URNs). This means that so far the only option is
to assign skills clusters via titles. Since it seems to be a good one, I say
it's fine (but it's a limitation of the script - as it is, it won't allow for
easy to implement generalisations).

Finally, this script will actually only work for engineering NOS
'''

# In[1]:
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
    from collections import OrderedDict
    
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
    import zlib
    
    #from gensim.scripts.glove2word2vec import glove2word2vec
    
    
    # In[163]:
    
    
    from utils_bg import *
    from utils_nlp import *
    from utils_skills_clusters import load_and_process_clusters, skills_matches
    from utils_skills_clusters import public_skills, public_skills_full
    
    from map_NOS_to_pathways_utils import *
    #from map_NOS_to_requirements_utils import cols_v_clus, cols_v_occ, cols_v_occ_and_clus
    #from map_NOS_to_requirements_utils import SC_to_requirements
    
    #%%
    def plot_soc_distribution(final_nos, cluster_name_figs, KEY,
                              cluster_name_save, matched_oobsoc_to_soc2,SAVEFIG):
        #% Replot distribution of SOC codes for suites in this super-suite
        fig = plt.figure(figsize = (12,8))
        #tmp = final_nos['SOC4'].value_counts()
        soc_series = pd.Series(flatten_lol(final_nos['SOC4'].map(lambda x: [x] 
                        if isinstance(x,float) else x).values))
        tmp = soc_series.value_counts()
        tmp = tmp.iloc[:18].iloc[::-1]
        tmp.plot(kind = 'barh', color = nesta_colours[3])
        for ix,i in enumerate(tmp):
            if i>2:
                plt.text(i+2,ix-.1,'{}'.format(i), fontsize = 13)
        plt.xlabel('NOS counts',fontsize = 18)
        plt.ylabel('Occupations',fontsize = 18)
        plt.title('Occupations breakdown for \'{}\''.format(cluster_name_figs),
                  fontsize = 18)
        # substitute labels
        T = plt.yticks()
        for t in T[1]:
            x = int(t.get_text()[:-2])
            if x in matches_oobsoc_to_soc2:
                x = matches_oobsoc_to_soc2[x]
            k = socnames_dict[x]
            try:
                t.set_text(k.lower().capitalize())
            except:
                1
                #print(t)
        t=plt.yticks(T[0],T[1])
        plt.tight_layout()
        #print(tmp)
        if SAVEFIG:
            plt.savefig(output_dir + '/supersuites/NOS_occupations_for_{}_{}_v2.png'.format(
                    cluster_name_save,KEY), bbox_inches='tight')
            plt.close(fig)
        
    def heatmap_qual_exp(final_nos, cluster_name_figs, KEY,
                              cluster_name_save,SAVEFIG):
        hm2plot = pd.crosstab(final_nos['myExp-peak'], final_nos['myEdu-peak'],
                           normalize = 'all')
        hm2plot['Postgraduate'] = 0.0
        hm2plot = hm2plot[['Pregraduate','Graduate','Postgraduate']]
        #hm2plot = hm2plot.applymap(lambda x: int(x*100))
        fig, _= plotHM(hm2plot, normalize = False, 
                   title = 'Experience vs education requirements \n (\'{}\')'.format(
                                           cluster_name_figs), w=5, h=5)
        if SAVEFIG:
            plt.savefig(output_dir + '/supersuites/NOS_matchbysoc_exp_vs_edu_for_{}_{}_v2.png'.format(
                    cluster_name_save,KEY), bbox_inches='tight')
            plt.close(fig)
         
            skillsdf = extract_top_skills(final_nos,all_exp,all_edu,N=30)
            # mark skills that are not "public"
            skillsdf['is_public'] = skillsdf['skill'].map(lambda x: 
                replace_skills_with_public(x,public_skills,skills_matches))#x in public_skills)
            skillsdf['is_public_full'] = skillsdf['skill'].map(lambda x: 
                replace_skills_with_public(x,public_skills_full,skills_matches))#x in public_skills)_full
            # save
            skillsdf.to_csv(output_dir + '/supersuites/NOS_topskills_for_{}_{}_v2_public.csv'.format(
                    cluster_name_save,KEY))
                
    #%%
    def salary_by_suite(final_nos, cluster_name_figs, KEY,
                              cluster_name_save,larger_suites,SAVEFIG):
        # 2. Salary box plots for the biggest suites in engineering
        salary_by_suite = {}
        for suite in larger_suites:
            A = final_groups.get_group(suite)['Salary']
            A = A.values
            A = [t for t in list(A) if len(t)]
            A = [t for t in A if not isinstance(t,str)]
            A = np.concatenate(A)
            A = A[~np.isnan(A)]
            salary_by_suite[suite.capitalize()] = A
            
        t0 = time.time()
        df = pd.DataFrame({k:pd.Series(v) for k,v in salary_by_suite.items()})
        print_elapsed(t0, 'make dataframe for salaries')
        
        fig, ax = plt.subplots(figsize = (12,8))
        sns.boxplot(data=df, palette = [nesta_colours[t] 
                                    for t in [0,1,3,4,5,6,8,9]],
                    whis = .75, showfliers = False)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")
        plt.ylabel('Salary', fontsize = 18)
        plt.xlabel('Suite', fontsize = 18)
        plt.title('Salary distribution across NOS in \'{}\' suites'.format(
                cluster_name_figs), fontsize = 18)
        plt.tight_layout()
        if SAVEFIG:
            plt.savefig(output_dir + 
                '/supersuites/NOS_matchbysoc_full_salary_by_suites_for_{}_{}_v2.png'.format(
                            cluster_name_save,KEY), bbox_inches='tight')
            plt.close(fig)
        
    #%%
    def skills_clusters_distribution(final_nos, cluster_name_figs, KEY,
                              cluster_name_save,SAVEFIG):
        # distribution of skill clusters in the NOS belonging to each super-suite
        fig = plt.figure(figsize = (14,7))
        cluster_series = pd.Series(flatten_lol(final_nos['best_cluster_nos'].map(lambda x: [x] 
                        if isinstance(x,str) else x).values))
        tmp = cluster_series.value_counts()
        tmp.plot('bar', color= nesta_colours[3])
        for ix,i in enumerate(tmp):
            if i>2:
                plt.text(ix-.2, i+4,'{}'.format(i), fontsize = 13)
        plt.setp(plt.gca().get_xticklabels(), rotation=60, ha="right",
                     rotation_mode="anchor")
        plt.ylabel('NOS counts', fontsize = 18)
        plt.xlabel('Skills cluster', fontsize = 18)
        plt.title('Distribution of skills cluster across NOS (\'{}\')'.format(
                    cluster_name_figs), fontsize = 18)
        plt.tight_layout()
        if SAVEFIG:
            plt.savefig(output_dir + '/supersuites/NOS_skills_clusters_for_{}_{}_v2.png'.format(
                    cluster_name_save,KEY))
            plt.close(fig)
        
    #%% distribution of both SC and SOC
    def heatmap_soc_skills(final_nos, cluster_name_figs, KEY,
                              cluster_name_save,SAVEFIG):
        cluster_series = pd.Series(flatten_lol(final_nos['best_cluster_nos'].map(lambda x: [x] 
                        if isinstance(x,str) else x).values))
        soc_series = pd.Series(flatten_lol(final_nos['SOC4'].map(lambda x: [x] 
                        if isinstance(x,float) else x).values))
        hm2plot = pd.crosstab(soc_series.map(lambda x : socnames_dict[x]),
                                  cluster_series).T
        hm2plot = hm2plot/hm2plot.sum()
        h1, w1 = hm2plot.shape
        fig, _ =plotHM(hm2plot, normalize = False, w = .5*w1+6, h = .5*h1+2,
                   title = 'Occupations vs skills clusters (\'{}\')'.format(
                                           cluster_name_figs))
        plt.tight_layout()
        if SAVEFIG:
            plt.savefig(output_dir + '/supersuites/NOS_SC_vs_SOC_for_{}_{}_v2.png'.format(
                    cluster_name_save,KEY))
            plt.close(fig)
            
    #%%
    def socs_per_suite(final_nos, cluster_name_figs, KEY,
                              cluster_name_save,SAVEFIG):
        # how many SOCs per suite?
        nb_of_socs = []
        for name,group in final_groups:
            print(name)
            tmp = pd.Series(flatten_lol(group['SOC4'].map(lambda x: [x] 
                        if isinstance(x,float) else x).values)).value_counts()
            print(tmp)
            nb_of_socs.append(len(tmp))
            print('-'*30)
        
        #% multi utility network construction (4), down stream gas (3)
        fig = plt.figure(figsize = (5,4))
        tmp=plt.hist(nb_of_socs, color = nesta_colours[3])
        for ix,i in enumerate(tmp[0]):
            if i>0:
                plt.text(tmp[1][ix], i+.5,'{}'.format(i), 
                         fontsize = 13)
        plt.ylabel('Suite counts', fontsize = 18)
        plt.xlabel('Number of occupations in a given suite', fontsize = 18)
        plt.title('Occupation variability \n within suites (\'{}\')'.format(
                    cluster_name_figs), fontsize = 18)
        plt.tight_layout()
        if SAVEFIG:
            plt.savefig(output_dir + '/supersuites/NOS_socs_per_suites_for_{}_{}_v2.png'.format(
                    cluster_name_save,KEY), bbox_inches='tight')
            plt.close(fig)
        
#%%
'''  Start of the main script   '''
#checkpoint
SETUP = True
if SETUP:    
    #%% set up main parameters
    #from set_params_thematic_groups import qualifier, qualifier0, pofs, WHICH_GLOVE, 
    #from set_params_thematic_groups import glove_dir, paramsn
    qualifier = 'postjoining_final_no_dropped'
    qualifier0 = 'postjoining_final_no_dropped'
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
    
    
    paramsn['bywhich'] = 'docs' #'docs' #'suites'
    paramsn['mode'] = 'tfidf' #'tfidf' #'meantfidf' #'combinedtfidf' #'meantfidf'

    # In[3]:
    
    # set up plot style
    print(plt.style.available)
    plt.style.use(['seaborn-darkgrid','seaborn-poster','ggplot'])
    
    #Get the NOS data 
    df_nos = pd.read_pickle(lookup_dir + 'all_nos_input_for_nlp_{}.zip'.format(qualifier0))
    
    # load the cleaned and tokenised dataset
    df_nos = df_nos.join(pd.read_pickle(lookup_dir + 
                        'all_nos_input_for_nlp_{}_pruned_{}.zip'.format(qualifier,pofs)))

    # remove p and k
    df_nos['pruned'] = df_nos['pruned'].map(remove_pk)
    print('Done')
    
    
    # Load stopwords
    with open(lookup_dir + 'stopwords_for_nos_{}_{}.pickle'.format(qualifier,pofs),'rb') as f:
        stopwords0, no_idea_why_here_stopwords, more_stopwords = pickle.load(f)
    stopwords = stopwords0 + no_idea_why_here_stopwords 
    stopwords += tuple(['¤', '¨', 'μ', 'บ', 'ย', 'ᶟ', '‰', '©', 'ƒ', '°', '„'])
    stopwords0 += tuple(['¤', '¨', 'μ', 'บ', 'ย', 'ᶟ', '‰', '©', 'ƒ', '°', '„',
                     "'m", "'re", '£','—','‚°','●'])
    stopwords0 += tuple(set(list(df_nos['Developed By'])))
    stopwords0 += tuple(['cosvr','unit','standard','sfl','paramount','tp','il',
                         'al','ad','hoc','someone','task','gnem','role',
                         'fin','flake','multiple','greet','go','find',
                         'agreed','agree','give','un','day','lloyd',
                         'whatever','whoever','whole','try','week','year','years',
                         'say','quo','1', '@','legacy','&','\uf0b7'])
    stopwords0 += tuple(['system','systems'])    
    
    # In[21]:
    # load which suites are in each super-suite
    super_suites_files=  ''.join(['/Users/stefgarasto/Google Drive/Documents/data/',
                                  'NOS_meta_data/NOS_Suite_Priority.xlsx'])
    super_suites_names = ['Engineering','Management','FinancialServices','Construction']
    all_super_suites, standard_labels, all_matches, all_match_names = match_super_suite_names(
                            df_nos,super_suites_names,super_suites_files)
    
    
    # In[32]:
    # assign supersuite and SOC codes
    df_nos['supersuite'] = df_nos['One_suite'].apply(lambda x: 
        assign_supersuite(x,all_match_names))
        
    # extract 2 digit soc
    df_nos['SOC4str'] = df_nos['Clean SOC Code'].map(adjustsoccode)
    df_nos['SOC1'] = df_nos['SOC4str'].map(extract1digits)
    df_nos['SOC2'] = df_nos['SOC4str'].map(extract2digits)
    df_nos['SOC3'] = df_nos['SOC4str'].map(extract3digits)
    df_nos['SOC4'] = df_nos['SOC4str'].map(extract4digits)
    print(df_nos['supersuite'].value_counts())
    
    #%%
    # load LSH groups and transferable NOS
    # load transferable NOS
    transferable_file = ''.join(['/Users/stefgarasto/Google Drive/Documents/results/NOS/nlp_analysis/',
                                'transferable_nos_n_postjoining_final_no_dropped/estimated_transferable_nos.csv'])
    transferable_nos = pd.read_csv(transferable_file)
    transferable_nos = transferable_nos.set_index('Unnamed: 0')
    transferable_nos['transferable'] = True

    # load duplicated NOS
    lshduplicate_file = ''.join(['/Users/stefgarasto/Google Drive/Documents/results/NOS/nlp_analysis/',
             'LSH_results_grouped_with_score_postjoining_final_no_dropped_th0.8.csv'])
    lshduplicate_nos = pd.read_csv(lshduplicate_file)
    
    #%% process the LSH duplicates to assign group ID to NOS
    # NOTE: these are all the groups, not just engineering NOS
    def split_nos_in_groups(x):
        if isinstance(x,str):
            x = [t.strip() for t in x.replace(')','').replace('(','').replace('\'','').split(',')]
            return x[-1]
        else:
            return x
    tmp0 = lshduplicate_nos.applymap(split_nos_in_groups)
    df_nos_lsh = tmp0[['Unnamed: 0','Avg group similarity','1']]
    t0 = time.time()
    for i in range(2, len(lshduplicate_nos.columns)-2):
        tmp = tmp0[['Unnamed: 0','Avg group similarity','{}'.format(i)]].rename(columns = {'{}'.format(i):'1'})
        tmp = tmp[tmp['1'].notna()]
        df_nos_lsh = pd.concat([df_nos_lsh, tmp])
    print_elapsed(t0, 'assigning LSH groups to NOS')

    #%% Join LSH groups and transferable NOS
    df_nos2 = df_nos.join(df_nos_lsh.rename(columns = {'Unnamed: 0': 'lsh_group', 
                                                      'Avg group similarity': 'lsh_simil',
                                                      '1':'index'}).set_index('index'), how = 'left')
    df_nos2 = df_nos2.join(transferable_nos[['transferable', 'avg similarity2','engineeringness',
                                                     'centralities2','we_spread']], how = 'left')
    df_nos2['lsh_group'].mask(df_nos2['lsh_group'].isnull(), 'na', inplace= True)
    # remove LSH groups with a low overall similarity
    th_lsh = 0.7
    df_nos2['lsh_group'].mask(df_nos2['lsh_simil']<th_lsh, 'na', inplace= True)
    df_nos2['transferable'].mask(df_nos2['transferable'].isnull(), False, inplace= True)
    
    #%% eliminate transferable NOS + NOS without SOC + legacy NOS
    # first remove legacy and nos without socs
    not_legacy_flag = (df_nos2['NOS Title'].map(lambda x: 'legacy' not in x)
                       ) & (df_nos2.index.map(lambda x: ~(x.endswith('l.pdf'))))
    with_soc_flag = df_nos2['SOC4'].notna()
    df_nos2 = df_nos2[not_legacy_flag & with_soc_flag]
    # now remove transferable NOS too
    df_nos3 = df_nos2[df_nos2['transferable'] == False]
    
    #%% combine LSH groups (things to combine: titles + SOCs + supersuite + suite)
    # then aggregate by lsh groups and combine by averaging/selecting (so that I don't have to create complicated
    # rules based on legacies, missing socs and transferability)
    def aggregate_floats_lsh(x):
        x = list(set(list(x)))
        #print(x.iloc[0])
        if len(x)==1:
            x = x[0]
        return x
    
    def aggregate_suites_lsh(x):
        #x = sum(list(x))
        print('!',x)
        return "{%s}" % ', '.join(x) #x = [x]
        #return 
    
    def aggregate_titles_lsh(x):
        # join the strings
        x = ';'.join(x)
        # remove duplicates and join again
        x = ';'.join(list(set(x.split(';'))))
        return x
    
    def aggregate_by_first(x):
        return x.iloc[0]
    
    def aggregate_supersuites(x):
        y=x.value_counts()
        if (len(y)==2) & ('other' in y.index):
            # return the more interesting supersuite
            y = [t for t in y.index if t!='other']
            return y[0]
        else:
            return pd.Series.mode(x)
                
    '''
    # Questions: 
    1. some LSH groups don't really have a similarity higher than 0.8. 
    Do I only want to merge those that do?
    Probably yes, but relax the threshold a bit because it's an average similarity 
    for groups with > 2 NOS (used 0.75)
    2. What to do with groups with more than one SOC when assigning requirements?
    3. How to assign skills cluster to groups? Just use the concatenated titles.
    
    # TOREMEMBERs: 
    1. check whether any SOC is not the same (g.SOC4.map(lambda x: len(x)).value_counts()): 3 rows have 2 SOCs.
    2. print which rows have two socs
    
    #TODOs:
    1. Check / change assignment of supersuite and suite!!
    '''
    #%% group details about engineering nos by custom/specific functions
    cols_of_interest = ['NOS Title','supersuite','One_suite','SOC4','SOC3',
                        'SOC2','SOC1',#'pruned_lemmas', 
                        'URN', 'lsh_group', 'lsh_simil', 
                        'transferable']# + list(svq_cols) + tfidf_cols + counts_cols
    
    # separate singletons NOS from grouped NOS
    df_nos_singles = df_nos3[df_nos3['lsh_group']=='na'][cols_of_interest]
    
    df_nos_grouped = df_nos3[df_nos3['lsh_group']!='na'][cols_of_interest
                                ].reset_index().groupby('lsh_group')
    agg_of_interest= {'URN': aggregate_floats_lsh, 'supersuite': aggregate_supersuites, 
                     'SOC4': aggregate_floats_lsh, 'index': aggregate_floats_lsh,
                     'SOC3': aggregate_floats_lsh, 'SOC2': aggregate_floats_lsh,
                     'SOC1': aggregate_floats_lsh, #'pruned_lemmas': sum,
                      'transferable': aggregate_by_first, #'lsh_group': aggregate_by_first,
                    'lsh_simil': np.mean}
    transform_of_interest = {'NOS Title': ','.join, 'One_suite': ','.join}
    
    t0 = time.time()
    #gm1 = df_nos_grouped[list(svq_cols)].agg(np.nanmean)
    #g0 = df_nos_eng_grouped[tfidf_cols].agg(np.nanmean)
    #g1 = df_nos_eng_grouped[counts_cols].agg(np.nanmean)
    g2 = df_nos_grouped[list(agg_of_interest.keys())].agg(agg_of_interest)#.reset_index()
    g3 = df_nos_grouped['One_suite'].apply(aggregate_titles_lsh)#.reset_index()
    g4 = df_nos_grouped['NOS Title'].apply(aggregate_titles_lsh)#.reset_index()
    df_nos_grouped = g2.join(g4, on = 'lsh_group').join(g3, on = 'lsh_group')#.join(
            #g1, on='lsh_group').join(g0, on = 'lsh_group').join(gm1, on = 'lsh_group')
    print_elapsed(t0,'aggregating LSH duplicates')
    
    # extract the columns of interest (minus lsh group) and concatenate single NOS and groups
    cols_of_interest = ['NOS Title','supersuite','One_suite','SOC4','SOC3','SOC2',
                        'SOC1', 'URN', 'lsh_simil', #'pruned_lemmas',
                        'transferable']# + list(svq_cols) + tfidf_cols + counts_cols
    df_nos4 = pd.concat([df_nos_singles[cols_of_interest], 
                         df_nos_grouped[cols_of_interest]])
    
    print('nb NOS x nb columns: ', df_nos4.shape)
    
    #%%
    #
    #1. Cycle through each supersuite to get aggregate requirements. Keep in mind
    #   that available skills clusters for engineering are different from the other
    #   supersuites
    supersuite_names = ['other', 'management', 'financialservices', 'construction', 'engineering']
    for supersuite in supersuite_names[:4]:
        #supersuite = 'engineering'
        df_nos_select = df_nos4[df_nos4['supersuite'].map(lambda x: supersuite in x)]
        
        #%%
        # create another column where the texts are lemmatised properly
        # This only allows working with titles
        def tag_titles(x):
            return nltk.pos_tag([t.strip().lower() for t in x.replace(';', ' ').split()])
            
        t0 = time.time()
        df_nos_select['pruned_lemmas'] = df_nos_select['NOS Title'].map(lambda x: 
            tidy_desc_with_pos(x,pofs_titles))
    #                 ).map(lambda x: lemmatise_pruned(x,pofs_titles))#pofs))
        print_elapsed(t0, 'lemmatising tagged tokens using only titles')        
        print(len(df_nos_select))
        
        #%%
        # define the transform: this one can easily be the same for both 
        # keywords and the clustering
        tfidf_n = define_tfidf(paramsn, stopwords0)
        
        # get the transform from the whole NOS corpus
        FULL_CORPUS = False
        if FULL_CORPUS:
            _, feature_names_n, tfidf_n, _ = get_tfidf_matrix(
                paramsn, df_nos4, tfidf_n, col = 'pruned_lemmas')
        else:
            _, feature_names_n, tfidf_n, _ = get_tfidf_matrix(
                paramsn, df_nos_select, tfidf_n, col = 'pruned_lemmas')
        
        
        
        # In[44]:    
        SAVEKW= False
            
        print('Number of features: {}'.format(len(feature_names_n)))
        N = 2000
        print('Some features:')
        print(feature_names_n[N:N+100])
        
        
        # In[77]:
        
        # first transform via tfidf all the NOS in one supersuite because you need the top keywords
        textfortoken = df_nos_select['pruned_lemmas']
        tfidfm = tfidf_n.transform(textfortoken)
    
        top_terms_dict = {}
        top_weights_dict = {}
        top_keywords_dict = {}
        #for name, group in ifa_df.groupby('Route'):
        igroup = 0
        n_keywords =[]
        n_repeated = []
        #top_terms = {}
        t0 = time.time()
        tfidfm_dense = tfidfm.todense()
        for ix,name in enumerate(df_nos_select.index):
            top_ngrams, top_weights, top_features = extract_top_features(
                    tfidfm_dense[ix,:], feature_names_n)
            if (len(top_features)==0) & USE_TITLES:
                # if no terms from the title survives the process - just add all terms
                top_features = list(df_nos_select.loc[name]['pruned_lemmas'])
                top_weights = [1]*len(top_features)
                
            top_terms_dict[name] = {}
            top_terms_dict[name] = top_features
            top_weights_dict[name] = {}
            top_weights_dict[name] = top_weights
            if ix<4:
                print(name, top_features) #, top_keywords)
                print('**************************************')
            
            #top_keywords, n1, n2  = get_top_keywords_nos(df_nos_select.loc[name]['Keywords'], stopwords0, top_n = 20)
            #top_keywords = [t for t in top_keywords if t != '-']
            #n_keywords.append(n1)
            #n_repeated.append(n2)
            #top_keywords_dict[name] = {}
            #top_keywords_dict[name] = top_keywords
            if ix % 1000 == 999:
                print('Got to NOS nb {}. Total time elapsed: {:.4f} s'.format(ix,time.time()-t0))
        # save them all as csv
        if SAVEKW and False:
            pd.DataFrame.from_dict(top_terms_dict, orient = 'index').to_csv(output_dir +
                                                        '/NOS_from_supersuites_top_terms_{}_{}.csv'.format(qualifier,pofs))
            pd.DataFrame.from_dict(top_weights_dict, orient = 'index').to_csv(output_dir +
                                  '/NOS_from_supersuites_top_terms_weights_{}_{}.csv'.format(qualifier,pofs))
        tfidfm_dense = None
        
        
        # In[82]:    
        # remove top terms that are not in the chosen gensim model
        new_top_terms_dict = {}
        new_top_weights_dict = {}
        for k,v in top_terms_dict.items():
            if len(v)==0:
                new_top_terms_dict[k] = v
                continue
            # check if the top terms for each document are in the gensim model
            if paramsn['ngrams']=='bi':
                new_top_terms, weights = prep_for_gensim_bigrams(v, model, 
                                                    weights = top_weights_dict[k])
            else:
                new_top_terms, weights = prep_for_gensim(v, model, 
                                                    weights = top_weights_dict[k])
            # only retains the ones in the model
            new_top_terms_dict[k] = new_top_terms
            new_top_weights_dict[k] = weights
            if np.random.randn(1)>3.5:
                print(k, new_top_terms, len(new_top_terms), len(v))
        
        
        
        #%% Up until here it should be the same as to when I get the thematic groups
        ''' Create skills clusters '''
        # create skill clusters
        if supersuite=='engineering':
            clus_names,comparison_vecs,skill_cluster_vecs =load_and_process_clusters(model,
                                                                            ENG=True)
        else:
            clus_names,comparison_vecs,skill_cluster_vecs =load_and_process_clusters(model,
                                                                            ENG=False)
        
        #%%
        feat_dict = {}
        feat_not_assigned = {}
        for feat in feature_names_n:
            #try:
            if len(feat.split())==1:
                if feat not in model:
                    feat_dict[feat]= []
                    continue
                feat_embedding = model[feat]            
            else:
                feat2 = feat.split()
                if any([elem not in model for elem in feat2]): #(feat2[0] not in model) | (feat2[1] not in model):
                    feat_dict[feat] = []
                    continue
                feat_embedding = sentence_to_vectors_nofile(feat,model)[0]
                #N = len(feat2)
                #feat_embedding = model[feat2[0]]/N
                #for elem in feat2[1:]:
                #    feat_embedding = feat_embedding + model[elem]/N
                
            feat_dict[feat]= highest_similarity_threshold_top_two(feat_embedding, 
                         comparison_vecs, clus_names, th= 0.4)
            if len(feat_dict[feat]) == 0:
                sims=cosine_similarity(feat_embedding.reshape(1,-1),comparison_vecs)
                possible_clus = highest_similarity_top_two(feat_embedding, 
                         comparison_vecs, clus_names)
                feat_not_assigned[feat]= [possible_clus,sims.max()]
    
        # get the inverse frequency of the features
        feat_df =pd.DataFrame.from_dict(feat_dict, orient='index')
        N = len(feature_names_n)
        clusters_idf = feat_df[0].value_counts().map(lambda x: np.log(N / x))
            
        
        #%%
        # ### Assign each NOS to a skill cluster
        
        '''
        Link each NOS to a skill cluster
        '''
        USE_WEIGHTED = False & (not USE_TITLES) & (not USE_KEYWORDS)
        st_v_clus = {}
        st_v_clus2 = {}
        full_clusters_counter = {}
        counter = 0
        for ix,k in enumerate(new_top_terms_dict):
            weights = 1
            # if no keywords, we can't make a decision
            if len(new_top_terms_dict[k])==0:
                st_v_clus[k] = ['uncertain']
                st_v_clus2[k] = ['uncertain']
                continue
            test_skills, full_test_skills  = get_mean_vec(new_top_terms_dict[k], model, 
                                                          weights = new_top_weights_dict[k])
            counts_clus = []
            weighted_clus_dict = {}
            for iskill in range(full_test_skills.shape[0]):
                top_clusters = highest_similarity_threshold_top_two(full_test_skills[iskill], 
                                                           comparison_vecs, clus_names,
                                                           th = 0.4)
                # to sum the importances of each keywords
                for iclus in top_clusters:
                    if iclus in weighted_clus_dict:
                        weighted_clus_dict[iclus] += new_top_weights_dict[k][iskill]
                    else:
                        weighted_clus_dict[iclus] = new_top_weights_dict[k][iskill]
                # to just get the counts
                counts_clus.append(top_clusters)
            counts_clus = flatten_lol(counts_clus)
            if len(counts_clus)==0:
                # no good keywords here, anywhere
                st_v_clus[k] = ['uncertain']
                st_v_clus2[k] = ['uncertain']
                continue
            # backup the skills cluster associated with the most important feature
            tmp_bkp= counts_clus[0]
            counts_clus = Counter(counts_clus).most_common()
            weighted_clus = Counter(weighted_clus_dict).most_common()
            full_clusters_counter[k] = [counts_clus, weighted_clus]
            
            # assign to a skills cluster
            # first case: only one skills clusters have been identified
            if len(counts_clus)==1:
                # easy
                st_v_clus[k] = counts_clus[0][0]
            # second case: there are just two skills clusters
            elif len(counts_clus)==2:
                # they occur an equal number of times
                if counts_clus[0][1]==counts_clus[1][1]:
                    # each has been identified only once: take them both
                    if counts_clus[0][1]==1:
                        st_v_clus[k] = [t[0] for t in counts_clus]
                    # each occur more than once
                    elif counts_clus[0][1]>1:
                        # can I break the tie using the tfidf weights?
                        if USE_WEIGHTED:
                            st_v_clus[k] = weighted_clus[0][0]
                        else:
                            # take them both
                            st_v_clus[k] = [t[0] for t in counts_clus]
                # they are not the same
                else:
                     #take the first one
                     st_v_clus[k] = counts_clus[0][0]
            # third case: multiple skills clusters identified, but just once each
            elif all([t[1]==1 for t in counts_clus]):
                st_v_clus[k] = highest_similarity_threshold_top_two(test_skills, comparison_vecs, 
                         clus_names, th = 0.3) #tmp_bkp
            
            # other cases: more than one cluster, at least one identified multiple times
            else:
                # should I use the weights?
                if USE_WEIGHTED:
                    st_v_clus[k] = weighted_clus[0][0]
                # are the first two the same?
                elif counts_clus[0][1]==counts_clus[1][1]:
                    # take them both
                    st_v_clus[k] = [t[0] for t in counts_clus[:2]]
                else:
                    # take the first one
                    st_v_clus[k] = counts_clus[0][0]
            if isinstance(st_v_clus[k],str):
                st_v_clus[k] = [st_v_clus[k]]
                
        #%%
        # add the best clusters to the nos dataframe
        tmp = pd.DataFrame.from_dict(st_v_clus, orient = 'index')
        tmp = tmp.rename(columns = {0: 'best_cluster_nos'})
        df_nos_select['best_cluster_nos'] = tmp['best_cluster_nos']
        
        existing_columns= ['NOS Title', 'supersuite', 'One_suite', 'SOC4', 'SOC3', 'SOC2', 'SOC1',
           'URN', 'lsh_simil', 'transferable', 'pruned_lemmas', 'best_cluster_nos']
        #%%
        # map NOS to requirements
        KEY = 'socs+clusters'
        final_nos = df_nos_select[existing_columns].join(SC_to_requirements(df_nos_select,
                                 KEY = KEY), how = 'left')
        df_nos_select = df_nos_select[existing_columns]
    
        #%% 
        #TODO: check this section
        ''' 
        save the new information about the NOS in super-suites
        For all of them, save the augmented dataframe (or just the new info, 
        that is without the full text)
        Well, let's say the important columns, that is:
        'NOS Title', 'URN', 'Original URN', 'Overview',
               'Knowledge_and_understanding', 'Performance_criteria', 'Scope_range',
               'Glossary', 'Behaviours', 'Skills', 'Values', 
               'Originating_organisation', 'Date_approved', 
               'Indicative Review Date', 'Version_number',
               'Links_to_other_NOS', 'External_Links', 'Developed By', 'Validity',
               'Keywords', 'NOS Document Status', 'NOSCategory',
               'Clean SOC Code', 'Occupations', 'Suite',
               'One_suite', 'All_suites', 'notes', 'extra_meta_info',
               'supersuite', 'best_cluster_nos', 'myExp', 'myEdu', 'Salary',
               'myExp-peak', 'myEdu-peak', 'Salary-peak', 'title_processed',
               'converted_skills', 'London'
        
        The dataframe to save is final_nos
        
        # possibly save as json?
        #Might take less space
        
        '''
        rel_cols = ['NOS Title', 'supersuite', 'One_suite', 'SOC4', 'SOC3', 'SOC2', 'SOC1',
           'URN', 'lsh_simil', 'transferable', 'pruned_lemmas', 'best_cluster_nos',
           'myExp', 'myEdu', 'Salary', 'myExp-peak', 'myEdu-peak', 'Salary-peak',
           'title_processed', 'converted_skills', 'London']
        SAVEAUG = False
        if SAVEAUG:
            final_nos[rel_cols].to_csv(output_dir + 
                         '/augmented_info_NOS_in_supersuites_{}.csv'.format(supersuite))
            tmp_save = output_dir + \
                    '/augmented_tfidf_NOS_in_supersuites_{}.pickle'.format(supersuite)
            with open(tmp_save,'wb') as f:
                pickle.dump((feature_names_n,tfidf_n),f)
            tmp_save = output_dir + \
                    '/augmented_top_features_NOS_in_supersuites_{}.gz'.format(supersuite)
            with open(tmp_save, 'wb') as fp:
                fp.write(zlib.compress(pickle.dumps((new_top_terms_dict, 
                                    new_top_weights_dict), pickle.HIGHEST_PROTOCOL),9))
                
        
        #%%
        # checkpoint
        DO_SUPERS = False
        SAVEFIG = True
        #% first, plots for all supersuites
        if DO_SUPERS:
            SELECT_MODE = supersuite
        #for SELECT_MODE in ['financialservices']: #['engineering','construction',
        #'management','financialservices']:
            
            #%
            '''
            Figure plotting for each supersuite.
            
            1. Heatmap of qualification vs experience requirements for the whole of the
            super-suite
            2. Salary box plots for the biggest suites in the super-suite
            3. Distribution of skills clusters per NOS across the super-suites (?)
            
            TODO: for management and construction, some skills clusters appear as letters
            '''
            
            ## get the data for this suite
            _, final_groups, larger_suites, cluster_name, cluster_name_save, \
                cluster_name_figs = select_subdf(SELECT_MODE, [], 
                                                 [], final_nos)
     
            # replace oob soc codes
            final_nos['SOC4'] = final_nos['SOC4'].map(replace_oob_socs)

            
            #%            
            plot_soc_distribution(final_nos, cluster_name_figs, KEY,
                              cluster_name_save, matched_oobsoc_to_soc2)
            
            #%
            # 1. Heatmap of qualification vs experience requirements for the whole of the
            # engineering super-suite
            heatmap_qual_exp(final_nos, cluster_name_figs, KEY,
                              cluster_name_save)
            
            #%
            salary_by_suite(final_nos, cluster_name_figs, KEY,
                              cluster_name_save,larger_suites)
            
            #%
            skills_clusters_distribution(final_nos, cluster_name_figs, KEY,
                              cluster_name_save)
            
            #%
            heatmap_soc_skills(final_nos, cluster_name_figs, KEY,
                              cluster_name_save,SAVEFIG)
            
            #%
            socs_per_suite(final_nos, cluster_name_figs, KEY,
                              cluster_name_save,SAVEFIG)

#%%
# ### Collect some examples (this is for ad-hoc requests for the report)

# In[244]:

# this is to populate example tables with distributions
PRINTTABLE = False
if PRINTTABLE:
    eng_groups.get_group('aeronautical engineering suite 3')[
        ['NOS Title','best_cluster_nos','Exp3-peak','Eduv2-peak','MeanSalary-peak']]
    
    
    # In[255]:
    
    for s_clus in ['welding and machining','electrical engineering',
                   'driving and automotive maintenance']:
        print(s_clus)
        for key in ['myExp','myEdu']:
            total = sum(cols_v_clus[s_clus][key].values(), 0.0)
            for key2 in cols_v_clus[s_clus][key]:
                print(key2, cols_v_clus[s_clus][key][key2]/total)
            print('-'*30)
            
        print('*'*90)
    
    # In[ ]:
    for soccode in [2122.0, 8114.0, 8126.0]:
        print(soccode)
        for key in ['myExp','myEdu']:
            total = sum(cols_v_occ[soccode][key].values(), 0.0)
            for key2 in cols_v_occ[soccode][key]:
                print(key2, cols_v_occ[soccode][key][key2]/total)
            print('-'*30)
            
        print('*'*90)
    #for key in ['Exp3','Eduv2']:
    #    print(cols_v_clus[8114][key]/cols_v_clus['electrical engineering'][key].sum())
    #    print('-'*30)
    
    #%%
    for soccode in [2122.0, 8114.0, 8126.0]:
        for s_clus in ['welding and machining','electrical engineering',
                   'driving and automotive maintenance']:
            name0 = str(soccode) + '+' + s_clus
            print(name0)
            for key in ['myExp','myEdu']:
                try:
                    total = sum(cols_v_occ_and_clus[name0][key].values(), 0.0)
                    for key2 in cols_v_occ_and_clus[name0][key]:
                        print(key2, cols_v_occ_and_clus[name0][key][key2]/total)
                except:
                    continue
                print('-'*30)
            print('*'*90)


