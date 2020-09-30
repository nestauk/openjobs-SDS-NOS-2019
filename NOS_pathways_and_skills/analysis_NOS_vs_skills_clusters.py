'''
Script for WP3.

It links all NOS to a skills cluster and computes some average stats per cluster.




'''

matches_dir = '/Users/stefgarasto/Google Drive/Documents/results/NOS/nos_vs_skills/nos_vs_taxonomy'
output_dir= matches_dir
#checkpoint
FIRST_RUN = False
if FIRST_RUN:
    print('Setting up all the necessary structures')
    #get_ipython().run_line_magic('matplotlib', 'inline')

    import os
    import numpy as np
    import pandas as pd
    import pickle
    from ast import literal_eval    
    import seaborn as sns
    from collections import Counter, OrderedDict
    import scipy
    from scipy.cluster.vq import whiten, kmeans
    import time
    
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import mpatches #Rectangle
    import matplotlib.pyplot as plt
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn import mixture
    
    import nltk
    
    import gensim
    import re
    from fuzzywuzzy import process, fuzz
    
    from utils_bg import *
    from utils_nlp import *
    from utils_skills_clusters import load_and_process_clusters, skills_matches, \
                                        public_skills, public_skills_full, \
                                        skills_taxonomy_full,tax_first_to_second, \
                                        tax_second_to_first, tax_second_to_third, \
                                        tax_third_to_second, bottom_layer, \
                                        public_skills_membership, \
                                        avgsalary_dict, prop_jobs_dict, growth_dict, \
                                        First_level_colours, first_level_colours
    
    from map_NOS_to_pathways_utils import *
    #from map_NOS_to_requirements_utils import cols_v_clus, cols_v_occ, cols_v_occ_and_clus
    #from map_NOS_to_requirements_utils import SC_to_requirements
    
    #%%
    # create tax_third_to_first
    tax_third_to_first = {}
    for third_cluster in tax_third_to_second.keys():
        tax_third_to_first[third_cluster] = tax_second_to_first[tax_third_to_second[
                third_cluster]]

    # we don't want clusters that are too small    
    clusters_to_exclude = [t for t in prop_jobs_dict if prop_jobs_dict[t]<0.001]
    clusters_to_exclude += ['uncertain'] 
    clusters_to_exclude += [t.capitalize() for t in prop_jobs_dict if prop_jobs_dict[t]<0.001]
    clusters_to_exclude += ['Uncertain'] 
    
    #%%
    def mvalue_counts(local_series):
        return pd.Series(flatten_lol(local_series.map(lambda x: [x] 
                        if isinstance(x,str) else x).values)).value_counts()
        
    #%%
    def plot_soc_distribution(final_nos, cluster_name_figs, KEY,
                              cluster_name_save, matched_oobsoc_to_soc2, 
                              plot_type = 'barh', SAVEFIG= False):
        #% Replot distribution of SOC codes for suites in this super-suite
        fig = plt.figure(figsize = (12,8))
        #tmp = final_nos['SOC4'].value_counts()
        soc_series = pd.Series(flatten_lol(final_nos['SOC4'].map(lambda x: [x] 
                        if isinstance(x,float) else x).values))
        tmp = soc_series.value_counts()
        tmp = tmp.iloc[:18].iloc[::-1]
        tmp.plot(kind = plot_type, color = nesta_colours[3])
        for ix,i in enumerate(tmp):
            if i>2:
                plt.text(i+2,ix-.1,'{}'.format(i), fontsize = 13)
        if plot_type == 'barh':
            plt.xlabel('NOS counts',fontsize = 18)
            plt.ylabel('Occupations',fontsize = 18)
        else:
            plt.ylabel('NOS counts',fontsize = 18)
            plt.xlabel('Occupations',fontsize = 18)

        plt.title('Occupations breakdown \n ({})'.format(cluster_name_figs),
                  fontsize = 18)
        # substitute labels
        if plot_type == 'barh':
            T = plt.yticks()
        else:
            T = plt.xticks()
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
        if plot_type == 'barh':
            t=plt.yticks(T[0],T[1])
        else:
            t=plt.xticks(T[0],T[1])
        plt.tight_layout()
        #print(tmp)
        if SAVEFIG:
            plt.savefig(output_dir + '/nos_vs_skillsclusters/NOS_occupations_for_{}_{}_v3.png'.format(
                    cluster_name_save,KEY), bbox_inches='tight')
            plt.close(fig)
        
        
    #%%
    def skills_clusters_distribution(final_nos, cluster_name_figs, KEY, cluster_name_save,
                              w = 7, h= 14, plot_type= 'barh', SAVEFIG = False,
                              level = 'tax_third_level'):
        # distribution of skill clusters in the NOS belonging to each super-suite
        fig = plt.figure(figsize = (w,h))
        cluster_series = pd.DataFrame(flatten_lol(final_nos[level].map(lambda x: [x] 
                        if isinstance(x,str) else x).values))
        tmp = cluster_series.value_counts()
        tmp = tmp[tmp.index.map(lambda x: x not in clusters_to_exclude)] #!='uncertain']
        if plot_type == 'barh':
            tmp = tmp[::-1]
        if 'bar' in plot_type:
            tmp.plot(plot_type, color= nesta_colours[3])
        else:
            
        for ix,i in enumerate(tmp):
            if i>2:
                if plot_type == 'barh':
                    plt.text(i+4, ix-.2,'{}'.format(i), fontsize = 13)
                else:
                    plt.text(ix-.2, i+4,'{}'.format(i), fontsize = 13)
        plt.setp(plt.gca().get_xticklabels(), rotation=60, ha="right",
                     rotation_mode="anchor")
        if plot_type=='barh':
            plt.xlabel('NOS counts', fontsize = 18)
            plt.ylabel('Skills cluster', fontsize = 18)
            T = plt.yticks()
            for t in T[1]:
                t.set_text(t.get_text().lower().capitalize())
            t=plt.yticks(T[0],T[1])
        else:
            plt.ylabel('NOS counts', fontsize = 18)
            plt.xlabel('Skills cluster', fontsize = 18)
            T = plt.xticks()
            for t in T[1]:
                t.set_text(t.get_text().lower().capitalize())
            t=plt.xticks(T[0],T[1])
            
        plt.title('Distribution of skill clusters across NOS \n ({})'.format(
                    cluster_name_figs), fontsize = 18)
        plt.tight_layout()
        if SAVEFIG:
            plt.savefig(output_dir + '/nos_vs_skillsclusters/NOS_skills_clusters_for_{}_{}_v3.png'.format(
                    cluster_name_save,KEY))
            plt.close(fig)

       
    #%% distribution of both SC and SOC
    def heatmap_soc_skills(final_nos, cluster_name_figs, KEY,
                              cluster_name_save,SAVEFIG):
        cluster_series = pd.Series(flatten_lol(final_nos['tax_third_level'].map(lambda x: [x] 
                        if isinstance(x,str) else x).values))
        soc_series = pd.Series(flatten_lol(final_nos['SOC4'].map(lambda x: [x] 
                        if isinstance(x,float) else x).values))
        hm2plot = pd.crosstab(soc_series.map(lambda x : socnames_dict[x]),
                                  cluster_series).T
        hm2plot = hm2plot/hm2plot.sum()
        h1, w1 = hm2plot.shape
        fig, _ =plotHM(hm2plot, normalize = False, w = .5*w1+6, h = .5*h1+2,
                   title = 'Occupations vs skills clusters \n ({})'.format(
                                           cluster_name_figs))
        plt.tight_layout()
        if SAVEFIG:
            plt.savefig(output_dir + '/nos_vs_skillsclusters/NOS_SC_vs_SOC_for_{}_{}_v3.png'.format(
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
        plt.title('Occupation variability within suites \n ({})'.format(
                    cluster_name_figs), fontsize = 18)
        plt.tight_layout()
        if SAVEFIG:
            plt.savefig(output_dir + '/nos_vs_skillsclusters/NOS_socs_per_suites_for_{}_{}_v3.png'.format(
                    cluster_name_save,KEY), bbox_inches='tight')
            plt.close(fig)#!/usr/bin/env python3

        
    #%%
    def top_values_per_cluster(final_nos,COL,tax_level,tax_cluster_name,
                               N=5):
        local_series = final_nos[COL][final_nos[tax_level]==tax_cluster_name]
        if COL =='Keywords':
            local_series = local_series.map(lambda x:
                [t for t in get_top_keywords_nos(x,stopwords0)[0] if t!='-'])
            counts = mvalue_counts(local_series)
        elif COL == 'Occupations':
            local_series = local_series.map(lambda x: ' '.join(x) if isinstance(x,list) else x)
            local_series = local_series[local_series.map(lambda x: isinstance(x,str))]
            local_series = local_series.map(lambda x: [t.strip().lower() for t in x.split(';')])
            local_series = local_series.map(lambda x: [t for t in x if len(t)])
            counts = mvalue_counts(local_series)
        elif COL == 'tokens':
            local_series = local_series.map(lambda x:x.split(';'))
            counts = mvalue_counts(local_series)
        elif COL == 'pruned_lemmas':
            counts = mvalue_counts(local_series)
        elif COL in ['Relevant_occupations']:
            raise ValueError('This column should not be selected')
        else:
            counts = local_series.value_counts()
        N = np.min([len(counts),N])
        return counts.iloc[:N]

    def add_text_to_hist_new(values, xvalues = None, addval = None, orient = 'vertical'):
        if addval is None:
            addval = .5 + np.floor(2*np.log(max(values)))
        addx = -.2 if orient=='horizontal' else 0
        for ix,i in enumerate(values):
            if i>-1:
                if xvalues is None:
                    x = ix - .2
                else:
                    x = xvalues[ix] +.02
                if orient == 'vertical':
                    plt.text(i+addval, x, '{}'.format(i), fontsize = 14)
                else:
                    plt.text(x, ix+addval,'{}'.format(i), fontsize = 14)
    
    # capitalise labels
    def capitalise_labels(T):
        for t in T[1]:
            t.set_text(t.get_text().capitalize())
        return T

    
#%%
'''  Start of the main script   '''
#checkpoint
SETUP = False
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
    df_nos['Keywords'] = df_nos['Keywords'].map(lambda x: get_keywords_list(x,[]))
    df_nos.Occupations = df_nos.Occupations.map(lambda x: get_keywords_list(x,[]))
    df_nos.Occupations = df_nos.Occupations.map(lambda x: x if isinstance(x,list) else [])
    print(df_nos['supersuite'].value_counts())
    
    #%%
    # load LSH groups and transferable NOS
    # load transferable NOS
    #transferable_file = ''.join(['/Users/stefgarasto/Google Drive/Documents/results/NOS/nlp_analysis/',
    #                            'transferable_nos_n_postjoining_final_no_dropped/estimated_transferable_nos.csv'])
    #transferable_nos = pd.read_csv(transferable_file)
    #transferable_nos = transferable_nos.set_index('Unnamed: 0')
    #transferable_nos['transferable'] = True

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
    #df_nos2 = df_nos2.join(transferable_nos[['transferable', 'avg similarity2','engineeringness',
    #                                                 'centralities2','we_spread']], how = 'left')
    #df_nos2['transferable'].mask(df_nos2['transferable'].isnull(), False, inplace= True)

    df_nos2['lsh_group'].mask(df_nos2['lsh_group'].isnull(), 'na', inplace= True)
    
    # remove LSH groups with a low overall similarity
    th_lsh = 0.75
    df_nos2['lsh_group'].mask(df_nos2['lsh_simil']<th_lsh, 'na', inplace= True)
    
    #%% eliminate legacy NOS
    # do NOT remove nos without socs
    not_legacy_flag = (df_nos2['NOS Title'].map(lambda x: 'legacy' not in x)
                       ) & (df_nos2.index.map(lambda x: ~(x.endswith('l.pdf'))))
    df_nos2 = df_nos2[not_legacy_flag]
    # now remove transferable NOS too
    #df_nos3 = df_nos2[df_nos2['transferable'] == False]
    
    
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
        if any([isinstance(t,float) for t in x]):
            x = [str(t) for t in x]
        # join the strings
        try:
            x = ';'.join(x)
        except:
            print(x)
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
    
    #def aggregate_full_text(x):
        
    #%%    
    '''
    # Notes: 
    1. I'm only merging LSH groups with a similarity higher than 0.75.
    2. How to assign skills cluster to groups? Just use the concatenated titles.
    3. For the full text: concatenate the full texts as well - this way a skill
        is found if it appears in any of these texts.
    '''
    cols_of_interest = ['NOS Title', 'supersuite', 'One_suite', 'SOC4', 'URN',
                        'lsh_group', 'lsh_simil', 'clean_full_text',
                           'Originating_organisation', 'Developed By', 'Keywords',
                           'Occupations']
                           #'Date_approved_year',
                           #'Clean Ind Review Year', 'Version_number',
                           #'Validity', 'NOS Document Status', 'Status',
                           #'NOSCategory', 'Relevant_occupations']
                           
    '''['NOS Title','supersuite','One_suite','SOC4',
                        'URN', 'lsh_group', 'lsh_simil', 'clean_full_text']#, 
                        #'transferable']'''
    
    # separate singletons NOS from grouped NOS
    df_nos_singles = df_nos2[df_nos2['lsh_group']=='na'][cols_of_interest]
    
    df_nos_grouped = df_nos2[df_nos2['lsh_group']!='na'][cols_of_interest
                                ].reset_index().groupby('lsh_group')
    agg_of_interest= {'URN': aggregate_floats_lsh, 'supersuite': aggregate_supersuites, 
                     'SOC4': aggregate_floats_lsh, 'index': aggregate_floats_lsh,
                     'lsh_simil': np.mean, 'clean_full_text': sum, 
                     'Keywords':sum, 'Occupations':sum}
                      #'transferable': aggregate_by_first, #'lsh_group': aggregate_by_first,
                    
    transform_of_interest = {'NOS Title': ','.join, 'One_suite': ','.join}
    
    t0 = time.time()
    #g1 = df_nos_grouped['clean_full_text'].agg(sum)
    g2 = df_nos_grouped[list(agg_of_interest.keys())].agg(agg_of_interest)
    g3 = df_nos_grouped['One_suite'].apply(aggregate_titles_lsh)
    g4 = df_nos_grouped['NOS Title'].apply(aggregate_titles_lsh)
    g4 = pd.DataFrame(g4)
    for col in ['Originating_organisation','Developed By']:
        g4 = g4.join(pd.DataFrame(df_nos_grouped[col].apply(aggregate_titles_lsh)),
                     on = 'lsh_group')
        
    #%%
    df_nos_grouped = g2.join(g4, on = 'lsh_group').join(g3, on = 'lsh_group')

    # one small adjustment: select first supersuite for those NOS that span
    # more than one
    df_nos_singles['one_supersuite'] = df_nos_singles['supersuite']
    df_nos_grouped['one_supersuite'] = df_nos_grouped['supersuite'].map(lambda x: x[0] 
                if isinstance(x,np.ndarray) else x)
    print_elapsed(t0,'aggregating LSH duplicates')
    
    #%% group details about engineering nos by custom/specific functions
    # finally remove NOS that belong to an LSH group
    
    #df_nos4 = df_nos2[df_nos2['lsh_group']=='na']#[cols_of_interest]
    cols_of_interest_new = [t for t in cols_of_interest if t !='lsh_group'] + ['one_supersuite']#, 
                        #'transferable']
    df_nos4 = pd.concat([df_nos_singles[cols_of_interest_new], 
                         df_nos_grouped[cols_of_interest_new]])
    print('nb NOS x nb columns: ', df_nos4.shape)
    
#%%
#checkpoint
COMPUTE_CLUSTERS= False
if COMPUTE_CLUSTERS:
    #1. Cycle through each supersuite to get aggregate requirements. Keep in mind
    #   that available skills clusters for engineering are different from the other
    #   supersuites
    supersuite_names = ['other', 'management', 'financialservices', 'construction', 
                        'engineering']
    feat_to_clusters = {}
    for iy,supersuite in enumerate(supersuite_names):
        #supersuite = 'engineering'
        df_nos_select = df_nos4[df_nos4['one_supersuite'].map(lambda x: supersuite in x)]
        
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
        #with open('tmp_feature_{}.pickle'.format(supersuite),'wb') as f:
        #    pickle.dump((feat_dict,feature_names_n,feat_not_assigned),f)
        # save them all as csv
        if SAVEKW and False:
            pd.DataFrame.from_dict(top_terms_dict, orient = 'index').to_csv(output_dir +
                                                        '/NOS_from_supersuites_top_terms_{}_{}_v3.csv'.format(qualifier,pofs))
            pd.DataFrame.from_dict(top_weights_dict, orient = 'index').to_csv(output_dir +
                                  '/NOS_from_supersuites_top_terms_weights_{}_{}_v3.csv'.format(qualifier,pofs))
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
        if supersuite in ['engineering','construction']:
            clus_names,comparison_vecs,skill_cluster_vecs =load_and_process_clusters(model,
                                                                            ENG=True)
        else:
            clus_names,comparison_vecs,skill_cluster_vecs =load_and_process_clusters(model,
                                                                            ENG=False)
        
        #%%
        feat_dict = {}
        feat_not_assigned = {}
        for counter,feat in enumerate(feature_names_n):
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
                        #highest_fuzzymatch(feat.replace('\'',''),
                             #                     skill_cluster_membership,
                             #                     th = 80, limit=3)
                        #highest_similarity_threshold_top_two(feat_embedding, 
                        # comparison_vecs, clus_names, th= 0.4)
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
        t0 = time.time()
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
                                            comparison_vecs, clus_names, th = 0.4)
                                #feat_dict[new_top_terms_dict[k][iskill]]
                               #highest_fuzzymatch(new_top_terms_dict[k][iskill],
                               #                   skill_cluster_membership,limit=3)
                               #
                if new_top_terms_dict[k][iskill] not in feat_to_clusters.keys():
                    feat_to_clusters[new_top_terms_dict[k][iskill]] = top_clusters
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
        print_elapsed(t0, 'computing association to skills clusters')
        
        #%%
        # add the best clusters to the nos dataframe
        tmp = pd.DataFrame.from_dict(st_v_clus, orient = 'index')
        tmp = tmp.rename(columns = {0: 'tax_third_level'})
        df_nos_select['tax_third_level'] = tmp['tax_third_level']
        for k in new_top_terms_dict.keys():
            new_top_terms_dict[k] = ';'.join(new_top_terms_dict[k])
        tmp = pd.DataFrame.from_dict(new_top_terms_dict, orient = 'index')
        tmp = tmp.rename(columns = {0: 'tokens'})
        df_nos_select['tokens'] = tmp['tokens']
        
        existing_columns= cols_of_interest_new + ['pruned_lemmas', 'tax_third_level', 'tokens']
        '''['NOS Title', 'supersuite', 'One_suite', 'SOC4', 'SOC3', 
                           'SOC2', 'SOC1', 'URN', 
                           'pruned_lemmas', 'tax_third_level', 'tokens',
                           'Originating_organisation', 'Date_approved', 'Date_approved_year',
                           'Clean Ind Review Year', 'Version_number',
                           'Developed By', 'Validity', 'Keywords',
                           'NOS Document Status', 'Status',
                           'NOSCategory', 'Occupations', 'OccupationsMetadata',
                           'Relevant_occupations','clean_full_text']'''
        
        #%%
        # map NOS to requirements
        #KEY = 'socs+clusters'
        #final_nos = df_nos_select[existing_columns].join(SC_to_requirements(df_nos_select,
        #                         KEY = KEY), how = 'left')
        #df_nos_select = df_nos_select[existing_columns]
        if iy == 0:
            df_nos = df_nos_select[existing_columns]
        else:
            df_nos = df_nos.append(df_nos_select[existing_columns])

    # remove NOS with no skills clusters
    df_nos2 = df_nos[df_nos['tax_third_level'].notna()]
    # assign second taxonomy level from third
    df_nos2['tax_second_level'] = df_nos2['tax_third_level'].map(lambda x:
        'uncertain' if x=='uncertain' else tax_third_to_second[x])
    # assign first taxonomy level from second
    df_nos2['tax_first_level'] = df_nos2['tax_second_level'].map(lambda x:
        'uncertain' if x=='uncertain' else tax_second_to_first[x])

      
#%%
#checkpoint
EXTRACT_SKILLS = False
if EXTRACT_SKILLS:
    #%%
    def check_bigrams(list_skills):
        already_there = []
        for t in list_skills:
            checks = [(t in item) and (t!=item) for item in list_skills]
            if any(checks):
                already_there.append((t, [item for ix,item in enumerate(list_skills) if checks[ix]]))      
        return already_there
    
    def correct_exact_matches(row_nos):
        text_nos = row_nos['clean_full_text']
        exact_matches = row_nos['exact_matches']
        duplications = row_nos['duplications']
        if len(duplications)==0:
            return exact_matches
        # manual removal of external links
        skills_to_remove = []
        for short_skill, long_skills in duplications:
            # t is a tuple: (short skill, long skill).
            # I want to check if 'short skill' appears alone without 'long skill'
            # Step 1. remove long skill
            for long_skill in long_skills:
                text_nos_reduced = text_nos.replace(long_skill, '')
            if not short_skill in text_nos_reduced:
                # if now I can't find the short skill, remove it
                skills_to_remove.append(short_skill)
        corrected_exact_matches = [t for t in exact_matches if t not in skills_to_remove]
        return corrected_exact_matches
    
    def subtract_exact_matches(row_nos):
        tmp = [t for t in row_nos['exact_matches'] if t not in row_nos['correct_exact_matches']]
        return tmp

    #% now load skills extracted
    
    '''
    with open(os.path.join(matches_dir,'all_exact_matches_emsi.pickle'),'rb') as f:
        all_exact_matches_emsi = pickle.load(f)
    with open(os.path.join(matches_dir,'all_exact_matches_nesta.pickle'),'rb') as f:
        all_exact_matches_nesta = pickle.load(f)
    with open(os.path.join(matches_dir,'all_fuzzy_matches_emsi.pickle'),'rb') as f:
        all_fuzzy_matches_emsi = pickle.load(f)
    with open(os.path.join(matches_dir,'all_fuzzy_matches_nesta.pickle'),'rb') as f:
        all_fuzzy_matches_nesta = pickle.load(f)
    with open(os.path.join(matches_dir,'indices_for_matches.pickle'),'rb') as f:
        indices_skills_extracted = pickle.load(f)
        
    # Get the skills match between Nesta and Emsi
    df_match_annotated = pd.read_csv(''.join(['/Users/stefgarasto/Google Drive/Documents/results/NOS/',
                     'nos_vs_skills/nesta_vs_emsi/nesta_to_emsi_bipartite_match_annotated.csv']))
    #df_match_annotated['good'][df_match_annotated['match']>94]='y'
    # add one row
    df_match_annotated = df_match_annotated.append(pd.DataFrame(
            data={'nesta':['supply chain'],'emsi': ['supply chain management'],
                  'match': [95],'good': ['y']}), sort=False)
    df_match_final = df_match_annotated[(df_match_annotated['match']>94) | 
            (df_match_annotated['good']=='y')].reset_index()
    df_match_final = df_match_final.set_index('nesta')
    
    # cross walk from Nesta to Emsi and join
    all_exact_matches = []
    for ix,item in enumerate(all_exact_matches_nesta):
        all_exact_matches.append(list(set([df_match_final.loc[t]['emsi'] for t in item] + 
                                        all_exact_matches_emsi[ix])))
    all_fuzzy_matches = []
    for ix,item in enumerate(all_fuzzy_matches_nesta):
        all_fuzzy_matches.append(list(set([df_match_final.loc[t]['emsi'] for t in item] + 
                                        all_fuzzy_matches_emsi[ix])))
    '''
    #%
    from extract_skills_from_nos import exact_skills_from_standards, fuzzy_skills_from_standards, \
        skills_matches_consensus2, skills_matches_consensus4, skills_lemmas
    #skills_matches_consensus4 = skills_matches_consensus4[
    #        ~skills_matches_consensus4.index.duplicated(keep='last')]
    #%%
    # find matches
    LOAD_MATCHES = False
    exists = os.path.isfile(matches_dir + '/exact_matches_in_nos.pickle')
    if LOAD_MATCHES and exists:
        with open(matches_dir + '/exact_matches_in_nos_with_emsi.pickle','rb') as f:
            all_exact_matches_nos,df_nos_indices = pickle.load(f)
    else:
        all_exact_matches_nos = exact_skills_from_standards(df_nos2, 'external')
        df_nos_indices = list(df_nos2.index)
        with open(matches_dir + '/exact_matches_in_nos_with_emsi.pickle','wb') as f:
            pickle.dump((all_exact_matches_nos,df_nos_indices),f)

    #exists = os.path.isfile(matches_dir + '/fuzzy_matches_in_nos.pickle')
    #if LOAD_MATCHES and exists:
    #    with open(matches_dir + '/fuzzy_matches_in_nos.pickle','rb') as f:
    #        all_fuzzy_matches,df_nos_indices_f = pickle.load(f)
    #    assert(df_nos_indices == df_nos_indices_f)
    #else:
    #    all_fuzzy_matches = fuzzy_skills_from_standards(df_nos2, 'external')
    all_fuzzy_matches = ['none']*len(all_exact_matches_nos)
    
    '''#If I need to remove some of the skills
    remove_list = ['client','secure','adapt','surface']
    all_exact_matches_nos = [[item for item in t if item not in remove_list]
        for t in all_exact_matches_nos]
    '''
    #%%
    # put all in a dataframe
    skills_extracted_df = pd.DataFrame(zip(all_exact_matches_nos,all_fuzzy_matches),
                                       index = df_nos_indices,
                                       columns= ['exact_matches','fuzzy_matches'])
    
    # check that all the indices are there
    assert(set(df_nos2.index)==set(skills_extracted_df.index))
    
    #%% Join to the main dataframe
    '''#if reloading:
    df_nos2 = df_nos2[['NOS Title', 'supersuite', 'One_suite', 'SOC4', 'URN', 
                       'lsh_simil',       'clean_full_text', 'Originating_organisation', 
                       'Developed By', 'Keywords', 'Occupations', 'one_supersuite', 
                       'pruned_lemmas',       'tax_third_level', 'tokens', 
                       'tax_second_level', 'tax_first_level']]
    '''
    df_nos2 = df_nos2.join(skills_extracted_df)
    
    #%% control for m-grams in n-grams where n>m
    df_nos2['duplications'] = df_nos2.exact_matches.map(check_bigrams)
    df_nos2['correct_exact_matches'] = df_nos2.apply(correct_exact_matches, axis =1)
    # manual removal of external link
    df_nos2['correct_exact_matches'] = df_nos2['correct_exact_matches'].map(
            lambda x: [t for t in x if t not in ['external link', 'external links']])
    df_nos2['differences'] = df_nos2.apply(subtract_exact_matches, axis = 1)
    
    #%% assign skills clusters
    skills_col2use = 'correct_exact_matches'
    df_nos2['exact_third_level'] = df_nos2[skills_col2use].map(
            lambda x: [skills_matches_consensus4.loc[t]['consensus_cluster'] for t in x])
    
    #%
    df_nos2['exact_second_level']= df_nos2['exact_third_level'].map(
            lambda x: [tax_third_to_second[t] for t in x])
    
    #%
    df_nos2['exact_first_level']= df_nos2['exact_second_level'].map(
        lambda x: [tax_second_to_first[t] for t in x])
    
    #%% get best-fit third cluster
    df_nos2['assigned_third'] = df_nos2.apply(lambda x: assign_third_cluster(x,level='third'), 
                               axis =1)
    df_nos2['assigned_second'] = df_nos2['assigned_third'].map(lambda x: 
                list(set([tax_third_to_second[t] if t!='uncertain' 
                      else 'uncertain' for t in x])))
    df_nos2['assigned_first'] = df_nos2['assigned_second'].map(lambda x: 
                list(set([tax_second_to_first[t] if t!='uncertain' 
                      else 'uncertain' for t in x])))
    df_nos2['direct_second'] = df_nos2.apply(lambda x: assign_third_cluster(x,level='second'),
                               axis =1)
    df_nos2['direct_first'] = df_nos2['direct_second'].map(lambda x: 
                list(set([tax_second_to_first[t] if t!='uncertain' 
                      else 'uncertain' for t in x])))
    
#%%       
def add_fl_colours_from_skills(entity_count,first_level_colours,tax_third_to_first,skills_matches_consensus4):
    clrs = list(entity_count.index.map(lambda x: first_level_colours[tax_third_to_first
                                [skills_matches_consensus4.loc[x]['consensus_cluster']]]))    
    patches = []
    for clr in first_level_colours.keys():
        if clr in clusters_to_exclude:
            continue
        patches.append(mpatches.Patch(color=first_level_colours[clr], label=clr.capitalize()))
    return patches, clrs

def add_fl_colours_from_clusters(entity_count,first_level_colours,tax_third_to_first):
    clrs = list(entity_count.index.map(lambda x: first_level_colours[tax_third_to_first[x]]))    
    patches = []
    for clr in first_level_colours.keys():
        if clr in clusters_to_exclude: #['Uncertain','uncertain']:
            continue
        patches.append(mpatches.Patch(color=first_level_colours[clr], label=clr.capitalize()))
    return patches, clrs

    
#%% checkpoint
EXAMINE_SKILLS = True
if EXAMINE_SKILLS:
    #skills_col2use = 'correct_exact_matches'
    
    #%% for how many did we (not) find any skills?
    skills_nb_count = df_nos2[skills_col2use].map(len)#.value_counts()
    f = plt.figure(figsize = (7,5))
    #skills_nb_count = skills_nb_count[::-1]#[-35:]
    #TODO: turn to histogram
    sns.distplot(skills_nb_count, color= nesta_colours[3], kde = False, norm_hist = False)
    avg_skill_nb = np.mean(skills_nb_count)
    plt.plot([avg_skill_nb,avg_skill_nb],[0,2900],'--',color=nesta_colours[3])
    plt.text(avg_skill_nb + 2, 2850, '{}'.format(np.around(avg_skill_nb,2)),
             fontdict={'size': 14})
    #add_text_to_hist_new(skills_count.values)
    plt.ylabel('Number of NOS', fontsize = 18)
    plt.xlabel('Number of skills extracted', fontsize = 18)
    plt.tight_layout()
    plt.savefig(os.path.join(matches_dir, 'number_of_skills_per_NOS_v3.png'))
    
    #%%
    ''' how many of the skills were found?'''
    skills_series = pd.DataFrame(flatten_lol(df_nos2[skills_col2use].values))
    # add first level clusters
    #skills_series['FL skill cluster'] = skills_series[0].map(lambda x: 
    #    tax_third_to_first[skills_matches_consensus4.loc[x]['consensus_cluster']].capitalize())
    skills_count = skills_series[0].value_counts()
    print('{} out of {} skills were found'.format(len(skills_count),
          len(skills_matches_consensus4)))
    
    ''' which are the most mentioned skills?'''
    f = plt.figure(figsize = (9,12))
    #skills_count = skills_count[::-1]
    clrs = list(skills_count.index.map(lambda x: first_level_colours[tax_third_to_first
                                [skills_matches_consensus4.loc[x]['consensus_cluster']]]))    

    #skills_count[-50:].plot('barh', color= nesta_colours[3])
    sns.countplot(y = 0, data=skills_series, order = skills_count.index[:50], palette = clrs[:50])
    patches = []
    for clr in First_level_colours.keys():
        if clr in clusters_to_exclude: #== 'Uncertain':
            continue
        patches.append(mpatches.Patch(color=First_level_colours[clr], label=clr))
    plt.legend(handles=patches, title = 'FL skill cluster', title_fontsize = 17)
    add_text_to_hist_new(skills_count[:50].values, xvalues= np.arange(.2,50.2))
    plt.xlabel('Number of occurrences', fontsize = 18)
    plt.ylabel('Skill', fontsize = 18)
    plt.title('Top extracted skills', fontsize = 20)
    T = plt.yticks()
    #for t in T[1]:
    #    t.set_text(skills_matches_consensus4.loc[t.get_text()]['original_skill'].capitalize())
    T = capitalise_labels(T)
    _ = plt.yticks(T[0],T[1])
    plt.tight_layout()
    plt.savefig(os.path.join(matches_dir, 'most_common_skills_extracted_v3.png'))
    
    #%%
    '''
    #% now also plot the top 50 bi-grams
    skills_count_two = skills_count[skills_count.index.map(lambda x: 
        len(x.split()))>1][-50:]
    f = plt.figure(figsize = (9,12))
    skills_count_two[-50:].plot('barh', color= nesta_colours[3])
    add_text_to_hist_new(skills_count_two[-50:].values)
    plt.xlabel('Number of occurrences', fontsize = 18)
    plt.ylabel('Skill', fontsize = 18)
    T = plt.yticks()
    #for t in T[1]:
    #    t.set_text(skills_matches_consensus4.loc[t.get_text()]['original_skill'].capitalize())
    T = capitalise_labels(T)
    _ = plt.yticks(T[0],T[1])
    plt.title('Top extracted skills', fontsize = 20)
    plt.tight_layout()
    plt.savefig(os.path.join(matches_dir, 'most_common_skills_extracted_bigrams_v3.png'))
    '''

    #%%
    #How similar are skills extracted to the keywords they have?
    # TODO (maybe)
    
    #%%
    # Just in case, separate search for the generic skills
    generic_skills = [['communication','communicating','communicate'],
                      ['writing','write'],
                      ['planning','plan'],
                      ['research'],
                      ['problem solving','problem solve'],
                      ['project management','project manage','project manager'],
                      ['self-starter','self start','self starter'],
                      ['customer service','serving customer','serving customers'],
                      ['computer skills','computer skill'],
                      ['budgeting','budget'],
                      ['teaching','teach'],
                      ['leadership','leader'],
                      ['mathematics','maths']]
    generic_skills_clusters = {'communication': [],
                               'writing':[],
                               'planning':[],
                               'research':[],
                               'problem solving': [],
                               'project management': [],
                               'self-starter': [],
                               'customer service': 'retail',
                               'computer skills': [],
                               'budgeting': [],
                               'teaching':'teaching',
                               'leadership': [],
                               'mathematics': 'physics and maths'}
    
    '''
    Possible other skills: 'self-starter','cust. service','computer skill'
    There are, but in more specific terms:
        budgeting, teaching, leadership, mathematics
    '''
    generic_skills_counts = {}
    for iskill in generic_skills:
        A = df_nos2['clean_full_text'].map(lambda x: any([t in x for t in iskill]))
        generic_skills_counts[iskill[0]] = sum(A)
                      
    print(generic_skills_counts)

    #%%
    '''Which third level clusters do extracted skills come from?'''
    Nplot = 50
    cluster_series = pd.DataFrame(flatten_lol(df_nos2['exact_third_level'].values))
    #cluster_series['first_level'] = cluster_series[0].map(lambda x: tax_third_to_first[x])
    tl_extracted = cluster_series[0].value_counts()
    f = plt.figure(figsize = (11,12))
    #tl_extracted = tl_extracted[::-1]
    patches, clrs = add_fl_colours_from_clusters(tl_extracted,first_level_colours,tax_third_to_first)
    
    #barlist = tl_extracted[-50:].plot('barh', color= nesta_colours[3])
    sns.countplot(y = 0, data=cluster_series, order = tl_extracted.index[:Nplot], palette = clrs[:Nplot])
    add_text_to_hist_new(tl_extracted[:Nplot].values,np.arange(.2,Nplot+.2))
    plt.legend(handles=patches, title = 'FL skill cluster', loc='lower right', title_fontsize = 16)
    
    plt.xlabel('Number of occurrences', fontsize = 18)
    plt.ylabel('Cluster', fontsize = 18)
    plt.title('Top extracted skill clusters',fontsize =20)
    T = plt.yticks()
    T = capitalise_labels(T)
    _ = plt.yticks(T[0],T[1])
    plt.tight_layout()
    plt.savefig(os.path.join(matches_dir, 'most_common_clusters_extracted_third_level_v3.png'))
    
    #%%
    print('The less common third level clusters are:')
    print(tl_extracted[:50])
    
    print('TL clusters that are never mentioned are: ')
    print(set(tax_third_to_second.keys()) - set(tl_extracted.index))
    
    '''
    #%%
    # let's see if I can show them all
    f = plt.figure(figsize = (7,27))
    plt.plot(tl_extracted,range(len(tl_extracted)))
    _ = plt.yticks(range(len(tl_extracted)), [t.capitalize() for t in tl_extracted.index], 
                   rotation='horizontal')
    plt.ylim([0,len(tl_extracted)])
    plt.tight_layout()
    plt.savefig(os.path.join(matches_dir, 'all_clusters_extracted_third_level_v3.png'))
    '''
    #%%
    '''
    # what's the distribution by first level skills cluster?
    TODO: rearrange the histogram so that it's colored by cluster
    
    clus_count = df_nos2['exact_first_level'].map(lambda x: Counter(x).most_common()[0][0].capitalize() 
        if len(x)>1 else None).value_counts()
    f = plt.figure(figsize = (8,6))
    clus_count[::-1].plot('barh', color= nesta_colours[3])
    add_text_to_hist_new(clus_count[::-1].values)
    plt.xlabel('Number of NOS', fontsize = 18)
    plt.ylabel('Skill cluster', fontsize = 18)
    plt.title('First taxonomy level')
    plt.tight_layout()
    plt.savefig(os.path.join(matches_dir, 'cluster_by_skills_extracted_first_level_v3.png'))
    #plt.close('all')
    '''
    
    #%%
    ''' now again but just counting how many times each cluster appear'''
    '''
    for level in ['first','second']:
        clus_count = pd.Series(flatten_lol(df_nos2['exact_{}_level'.format(level)
                                    ].values)).value_counts()
        f = plt.figure(figsize = (8,6)) if level == 'first' else plt.figure(figsize = (8,12))
        clus_count[::-1].plot('barh', color= nesta_colours[3])
        add_text_to_hist_new(clus_count[::-1].values)
        plt.xlabel('Number of occurrences', fontsize = 18)
        plt.ylabel('Skill cluster', fontsize = 18)
        plt.title('First taxonomy level')
        T = plt.yticks()
        T = capitalise_labels(T)
        plt.yticks(T[0],T[1])
        plt.tight_layout()
        plt.savefig(os.path.join(matches_dir, 'most_common_clusters_extracted_{}_level_v3.png'.format(
                level)))
    '''
    
        #%%
    '''Which clusters contain most of the skills not mentioned?'''
    missing_skills = list(set(skills_matches_consensus4.index).difference(skills_count.index))
    missing_clusters = skills_matches_consensus4.loc[[t for t in missing_skills]
                ]['consensus_cluster']
    missing_cluster_count = missing_clusters.value_counts()
    groups_length = pd.DataFrame(skills_matches_consensus4).groupby('consensus_cluster').agg(len)
    missing_cluster_count =pd.DataFrame(missing_cluster_count)
    missing_cluster_count['growth'] = missing_cluster_count.index.map(lambda x: 
                        growth_dict[x])
    missing_cluster_count['prop_jobs'] = missing_cluster_count.index.map(lambda x: 
                        prop_jobs_dict[x])
    print()
    print('Which clusters contain most of the skills not mentioned?')
    print(missing_cluster_count[:25])
    print()
    for t in missing_cluster_count.index:
        N= groups_length.loc[t]['lemmatised']#len(matches_grouped.get_group(t))
        missing_cluster_count.loc[t] = missing_cluster_count.loc[t]/N
    print('Now by percentage of skills missed')
    print(missing_cluster_count.sort_values('consensus_cluster')[::-1][:25])
    
    #%% as an example, print the missing skills for driving and welding
    A = missing_clusters[(missing_clusters.map(lambda x: x in 
            ['welding and machining','driving and automotive maintenance']))]
    A= pd.DataFrame(A.sort_values()).rename(columns = {'consensus_cluster':'Skill cluster'})
    A.index.rename('Skill name',inplace=True)
    pd.DataFrame(A).to_csv(os.path.join(matches_dir,'missing_skills_driving_and_welding_v3.csv'))
    
    
    #%%
    '''Cluster by cluster, check which skills are mentioned and which are not. 
    That is, what is the overlap between the skills mentioned in the groups of 
    NOS linked to each cluster and all the existing ones? Are there certain 
    skills that do not seem to be mentioned in the NOS dataset? Which groups of 
    skills are often forgotten?
    '''
    '''
    cluster_groups = df_nos2.groupby('tax_third_level')
    matches_grouped = pd.DataFrame(skills_matches_consensus4).groupby('consensus_cluster')
    extract_skills_membership = {}
    for name, g in cluster_groups:
        # take the top 20 skills for each cluster
        if name != 'uncertain':
            print('*'*100)
            print(name.capitalize())
            print('-'*10)
            top10skills = [t[0] #skills_matches_consensus4.loc[t[0]]['original_skill'] 
                for t in Counter(flatten_lol(g['exact_matches'].values)
                ).most_common()[:20]]
            consensus_skills = matches_grouped.get_group(name)
            intersect_top_skills = set(top10skills).intersection(consensus_skills.index)
            print(intersect_top_skills)
            extract_skills_membership[name] = {'top_extracted':top10skills,
                                     'top_intersection': list(intersect_top_skills),
                                     'public_skills' : public_skills_membership[name]}
    skills_compare_by_cluster = pd.DataFrame.from_dict(extract_skills_membership,
                                     orient ='index')
    #for col in ['top_extracted','top_intersection']:
    #    skills_compare_by_cluster[col]
    skills_compare_by_cluster.to_csv(os.path.join(matches_dir,'skills_extracted_by_assigned_cluster.csv'))
    #, columns = ['top_extracted']).join(
                                        #pd.DataFrame.from_dict(public_skills_membership, 
                                        #    orient ='index'))
    '''
    #%%
    '''
    Repeat the above suite by suite: most common skills and most common clusters
    mentioned
    
    '''
    suite_groups = df_nos2.groupby('One_suite')
    extract_skills_membership = {}
    for name, g in suite_groups:
        # take the top 20 skills for each cluster
        if len(g)<5:
            continue
        top10skills = [t[0].capitalize() #skills_matches_consensus4.loc[t[0]]['original_skill'] 
            for t in Counter(flatten_lol(g[skills_col2use].values)
            ).most_common()[:20]]
        top20clusters = [t[0].capitalize() for t in Counter(flatten_lol(g['exact_third_level'])).most_common()[:20]]
                #g['exact_cluster'].values)
        extract_skills_membership[name.capitalize()] = {'Top skills extracted':'; '.join(top10skills),
                                 'Corresponding top clusters': '; '.join(top20clusters),
                                 'Suite size': len(g)}
    skills_compare_by_suite = pd.DataFrame.from_dict(extract_skills_membership,
                                     orient ='index')
    skills_compare_by_suite = skills_compare_by_suite.sort_values('Suite size',
                                                                  ascending=False)
    skills_compare_by_suite.to_csv(os.path.join(matches_dir,'skills_extracted_by_suite_v3.csv'))
    
                 
    #%%
    def plot_distribution_from_df(df_nos, plotting_col, first_level_colours,tax_third_to_first,
                                  w=7, h=14, Nplot=50, x_label= 'Number of occurrences',
                                  y_label= 'Skill cluster', 
                                  title= 'Top extracted skill clusters',
                                  save_dir = output_dir, SAVEFIG = False,
                                  save_fig_name = 'my_figure.png'):
        
        cluster_series = pd.DataFrame(flatten_lol(df_nos[plotting_col].values))
        cluster_count = cluster_series[0].value_counts()
        cluster_count = cluster_count[cluster_count.index!='uncertain']
        f = plt.figure(figsize = (w,h))
        patches, clrs = add_fl_colours_from_clusters(cluster_count,first_level_colours,
                                                     tax_third_to_first)
        if Nplot is None:
            Nplot = len(cluster_count)
        sns.countplot(y = 0, data=cluster_series, order = cluster_count.index[:Nplot],
                      palette = clrs[:Nplot])
        add_text_to_hist_new(cluster_count[:Nplot].values,np.arange(.2,Nplot+.2))
        plt.legend(handles=patches, title = 'FL skill cluster', 
                   loc='lower right', title_fontsize = 16)
        
        plt.xlabel(x_label, fontsize = 18)
        plt.ylabel(y_label, fontsize = 18)
        plt.title(title, fontsize =20)
        T = plt.yticks()
        T = capitalise_labels(T)
        _ = plt.yticks(T[0],T[1])
        plt.tight_layout()
        if SAVEFIG:
            plt.savefig(os.path.join(save_dir, save_fig_name))
    
    plot_distribution_from_df(df_nos2,'assigned_third', first_level_colours, tax_third_to_first,
                                  w=10.5, h=26, Nplot=None, x_label= 'Number of occurrences',
                                  y_label= 'Skill cluster', 
                                  title= 'Top extracted skill clusters',
                                  save_dir = matches_dir + '/nos_vs_skillsclusters', 
                                  SAVEFIG = True,
                                  save_fig_name = 'NOS_skills_clusters_for__third_level_title_v3.png')
    
    #%%
    def plot_two_distribution_from_df(df_nos, plotting_col, comparison_dist,
                                      first_level_colours,tax_third_to_first,
                                      w=7, h=14, Nplot=50, x_label= 'Number of occurrences',
                                      y_label= 'Skill cluster', 
                                      title= 'Top extracted skill clusters',
                                      save_dir = output_dir, SAVEFIG = False,
                                      save_fig_name = 'my_figure.png'):
        
        cluster_series = pd.DataFrame(flatten_lol(df_nos[plotting_col].values))
        cluster_count = cluster_series[0].value_counts()
        a_vals = cluster_count[cluster_count.index!='uncertain']
        b_vals = comparison_dist.loc[[t for t in a_vals.index]]
        assert(all(a_vals.index==b_vals.index))
        f = plt.figure(figsize = (w,h))
        ax = plt.gca()
        patches, clrs = add_fl_colours_from_clusters(a_vals,first_level_colours,
                                                     tax_third_to_first)
        if Nplot is None:
            Nplot = len(a_vals)
            
        ind = np.arange(Nplot-1,-1,-1)
        #print(ind)
        width = 0.375
        
        # Set the colors
        
        def autolabel(bars):
            # attach some text labels
            for bar in bars:
                width0 = bar.get_width()
                ax.text(np.max([width0*0.99,0.2]), bar.get_y() + bar.get_height()/2,
                        '{}'.format(np.around(width0,1)),
                        ha='right', va='center')
        
        # make the plots
        a_vals = a_vals/a_vals.sum()*100
        #a_vals = a_vals[::-1]
        #b_vals = b_vals[::-1]
        #print(a_vals.values[:10],b_vals.values[:10])
        a = ax.barh(ind+ width, a_vals.values[:Nplot], width, color = clrs) # plot a vals
        b = ax.barh(ind, b_vals.values[:Nplot], width, color = clrs, alpha=0.5)  # plot b vals
        plt.ylim([-.5,Nplot])
        ax.set_yticks(ind + width)  # position axis ticks
        ax.set_yticklabels(a_vals.index[:Nplot])  # set them to the names
        
        patches.append(mpatches.Patch(color='k', label='Proportion of NOS'))
        patches.append(mpatches.Patch(color='k', alpha = 0.5, label='Proportion of jobs'))
        
        ax.legend(handles = patches, # + [a[0], b[0]], ['Proportion of NOS', 'Proportion of jobs'], 
                  loc='lower right')
        
        autolabel(a)
        autolabel(b)
        
    comparison_dist = pd.DataFrame.from_dict(prop_jobs_dict, orient='index',
                                           columns =['Proportion of jobs'])
    plot_two_distribution_from_df(df_nos2, 'assigned_third', comparison_dist['Proportion of jobs'],
                                      first_level_colours,tax_third_to_first,
                                      w=10.5, h=20, Nplot=50, x_label= 'Number of occurrences',
                                      y_label= 'Skill cluster', 
                                      title= 'Top extracted skill clusters',
                                  save_dir = matches_dir + '/nos_vs_skillsclusters', 
                                  SAVEFIG = True,
                                  save_fig_name = 'NOS_skills_clusters_for_third_level_title_compare.png')
    
    #%%
    ## distribution of NOS by third level skill clusters    
    #skills_clusters_distribution(df_nos2,'Third taxonomy level','third_level_title','',
    #                             h=27,w=10,SAVEFIG=True,level='assigned_third')
    #
    #skills_clusters_distribution(df_nos2,'Second taxonomy level','second_level_title','',
    #                             h=12,w=10,SAVEFIG=True,level='assigned_second')
    #
    #skills_clusters_distribution(df_nos2,'First taxonomy level','','first_level_title',
    #                             h=6,w=10,SAVEFIG=True,level='assigned_first')
    
    #%% Any cluster that is not mapped at all?
    print('*'*70)
    clusters_list1 = set(tax_third_to_second.keys())
    clusters_list2 = set(pd.Series(flatten_lol(df_nos2['assigned_third'])
                                                        ).value_counts().index)
    print('Nb of third level clusters never matched is {}'.format(
            len(clusters_list1-clusters_list2)))
    print('These clusters are:')
    print(clusters_list1 - clusters_list2)
    
    print()
    print('-'*30)
    clusters_list1 = set(tax_second_to_first.keys())
    clusters_list2 = set(pd.Series(flatten_lol(df_nos2['assigned_second'])
                                                            ).value_counts().index)
    print('Nb of second level clusters never matched is {}'.format(
            len(clusters_list1-clusters_list2)))
    print('These clusters are:')
    print(clusters_list1 - clusters_list2)
    print('*'*70)
    
    #%%
    numeric_df= pd.DataFrame.from_dict(growth_dict, orient='index',columns=['Growth']).join(
            pd.DataFrame.from_dict(avgsalary_dict, orient='index',columns=['bottom_salary',
                                                    'median_salary','top_salary'])).join(
                    pd.DataFrame.from_dict(prop_jobs_dict, orient='index',
                                           columns =['Proportion of jobs']))
    numeric_df = numeric_df.join(pd.DataFrame(pd.Series(flatten_lol(
            df_nos2['assigned_third'])).value_counts()))
    
    numeric_df = numeric_df.rename(columns = {0: 'Proportion of NOS'})
    
    numeric_df = numeric_df.join(pd.DataFrame(pd.Series(flatten_lol(
            df_nos2['exact_third_level'])).value_counts()))
    
    numeric_df = numeric_df.rename(columns = {0: 'Proportion of skills'})
    # only keep third level clusters
    numeric_df= numeric_df[numeric_df.index.map(lambda x: x in tax_third_to_second.keys())]
    
    numeric_df['FL skill cluster']= numeric_df.index.map(lambda x: 
                    tax_third_to_first[x].capitalize())
    # normalise
    for t in ['Proportion of NOS','Proportion of skills']:
        numeric_df[t] = numeric_df[t]/np.nansum(numeric_df[t])*100
    numeric_df['Proportion of NOS'][numeric_df['Proportion of NOS'].isnull()] = 0
    
    #%%
    # scatter plot, growth vs representation?
    plt.figure(figsize=(10,6))
    fig = sns.scatterplot(x = 'Growth', y = 'Proportion of NOS', data = numeric_df, 
                     hue = 'FL skill cluster', palette = First_level_colours,
                     alpha = .7)
    fig.legend(loc="upper left")#, bbox_to_anchor=(1.05, 1))
    plt.xlabel('Skill growth',fontsize = 18)
    plt.ylabel('Proportion of NOS',fontsize = 18)
    plt.tight_layout()
    high_growth = numeric_df[numeric_df['Growth']>1.8]
    for name,t in high_growth.iterrows():
        print(name)
        plt.scatter(t['Growth'],t['Proportion of NOS'],s=80,facecolors ='none',edgecolors='k')
        if 'flow' in name:
            plt.text(t['Growth']-.2,t['Proportion of NOS']-.25,name.capitalize(),fontsize = 11)
        elif 'assistance and care' in name:
            plt.text(t['Growth']+.02,t['Proportion of NOS']-.15,name.capitalize(),fontsize = 11)
        elif 'engineering' in name:
            plt.text(t['Growth']-.2,t['Proportion of NOS']+.2,name.capitalize(),fontsize = 11)
        elif 'dental assistance' in name:
            plt.text(t['Growth']+.02,t['Proportion of NOS']-.04,name.capitalize(),fontsize = 11)
        elif name in ['phlebotomy','data engineering','physiotherapy and beauty']:
            plt.text(t['Growth']+.02,t['Proportion of NOS']+.1,name.capitalize(),fontsize = 11)
        else:
            plt.text(t['Growth']+.02,t['Proportion of NOS'],name.capitalize(),fontsize = 11)
    for name in ['app development']:#,'web development']:#,'system administration','servers and middleware']:
        t = numeric_df.loc[name]
        plt.scatter(t['Growth'],t['Proportion of NOS'],s=80,
                    facecolors =First_level_colours[t['FL skill cluster']],edgecolors='k')
        plt.text(t['Growth']-.25,t['Proportion of NOS']-.25,name.capitalize(),fontsize = 11)
    plt.savefig(output_dir + '/nos_vs_skillsclusters/NOS_skills_clusters_vs_growth_v3.png')
    
    #%%
    numeric_df['ratio'] = numeric_df['Proportion of NOS']/numeric_df['Proportion of jobs']
    numeric_df.ratio[numeric_df.ratio==0] = np.nan
    A = numeric_df[['ratio','Proportion of jobs','Proportion of NOS','Growth']].sort_values(
            'ratio')
    # discard clusters with no actual representation in job adverts
    A = 
    A['log_ratio'] = A.ratio.map(np.log)
    #A = A[A.ratio.map(lambda x: x!=np.inf)]
    notna = A.ratio.map(lambda x: not np.isnan(x))
    #smallA = A.ratio[(abs(A.ratio)>.5) & (A['Proportion of jobs']>0.1) & 
    #                 (A['Proportion of NOS']>0)]
    smallA = A.log_ratio[abs(A.log_ratio)>np.log(3)]
    # 0.1% means roughly 50,000 job adverts
    # 0.001% means roughly 500 job adverts
    print(len(smallA))
    f = plt.figure(figsize = (7,12))
    smallA.plot('barh', color= nesta_colours[3])
    #barlist[0].set_color('r')
    #add_text_to_hist_new(tl_extracted[-50:].values)
    #plt.xlim([-1,10])
    plt.xlabel('Comparison ratio', fontsize = 17)
    plt.ylabel('Skill cluster', fontsize = 17)
    T = plt.yticks()
    T = capitalise_labels(T)
    _ = plt.yticks(T[0],T[1])
    plt.tight_layout()
    #plt.savefig(output_dir + '/nos_vs_skillsclusters/NOS_vs_jobs_representation_ratio_v3.png')
    
    #%% 
    print('comparison ratio for high growth clusters:')
    for name,t in high_growth.iterrows():
        print(name,A[name])
    
    print('-'*30)
    print('growth for highly underrepresented clusters:')
    for name in A[A>.5].index:
        print(name,numeric_df.loc[name]['Growth'])
    
#%%
# checkpoint
DO_OUTPUT = False
if DO_OUTPUT:
    '''
    Potentially interesting things:
    -Most commonly mapped clusters
    -Least commonly mapped clusters
    TODO: rearrange the histogram so that it's colored and ordered by 
    first level clusters
    TODO: remove "uncertain" cluster
    '''
    skills_clusters_distribution(df_nos2,'Third taxonomy level','third_level_title','',
                                 h=27,w=10,SAVEFIG=True)
    skills_clusters_distribution(df_nos2,'Second taxonomy level','','second_level_title',
                                 h=12,w=10,SAVEFIG=True,level='tax_second_level')
    skills_clusters_distribution(df_nos2,'First taxonomy level','','first_level_title',
                                 h=6,w=10,SAVEFIG=True,level='tax_first_level')

    #%% Any cluster that is not mapped at all?
    print('*'*70)
    clusters_list1 = set(tax_third_to_second.keys())
    clusters_list2 = set(df_nos2['tax_third_level'].value_counts().index)
    print('Nb of third level clusters never matched is {}'.format(
            len(clusters_list1-clusters_list2)))
    print('These clusters are:')
    print(clusters_list1 - clusters_list2)
    
    print()
    print('-'*30)
    clusters_list1 = set(tax_second_to_first.keys())
    clusters_list2 = set(df_nos2['tax_second_level'].value_counts().index)
    print('Nb of second level clusters never matched is {}'.format(
            len(clusters_list1-clusters_list2)))
    print('These clusters are:')
    print(clusters_list1 - clusters_list2)
    print('*'*70)
    
    #%%
    '''
    -Most commonly mapped clusters per suite. 
    -Perform a visual review of whether suites seem to correspond to the proper clusters, 
        which would be better associated with something else and which don’t really 
        have an ideal counterpart (on the saved pdf).
    '''
    list_of_suites = df_nos['One_suite'].value_counts()
    list_of_suites = list(list_of_suites[list_of_suites>5].index)
    suite_v_clus = {}
    for suite in list_of_suites:
        df_nos_select = df_nos2[df_nos2['One_suite']==suite]
        suite_v_clus[suite] = {}
        # third level
        tmp = mvalue_counts(df_nos_select['tax_third_level'])    
        suite_v_clus[suite]['third_level'] = tmp.index[0]#,tmp.iloc[0])
        # second level
        tmp = mvalue_counts(df_nos_select['tax_second_level'])    
        suite_v_clus[suite]['second_level'] = tmp.index[0]#,tmp.iloc[0])
        # first level
        tmp = mvalue_counts(df_nos_select['tax_first_level'])    
        suite_v_clus[suite]['first_level'] = tmp.index[0]#,tmp.iloc[0])
        # add the most common supersuite
        tmp = mvalue_counts(df_nos_select['supersuite'])
        suite_v_clus[suite]['supersuite'] = tmp.index[0]
    suite_v_clus_df = pd.DataFrame.from_dict(suite_v_clus, orient = 'index')
    suite_v_clus_df.to_csv(output_dir + 
                           '/nos_vs_skillsclusters/most_common_cluster_per_suite_by_title_v3.csv')
    
    #%% organise into groups
    suite_groups = suite_v_clus_df.groupby('second_level')
    suites_by_clus = {}
    
    for name,g in suite_groups:
        tmp = list(g.index)
        suites_by_clus[name] = tmp
    pd.DataFrame.from_dict(suites_by_clus, orient = 'index').T.to_csv(output_dir + 
                           '/nos_vs_skillsclusters/suites_organised_by_second_skills_cluster2_v3.csv')
    
    # organise into groups
    suite_groups = suite_v_clus_df.groupby('third_level')
    suites_by_clus = {}
    
    for name,g in suite_groups:
        tmp = list(g.index)
        suites_by_clus[name] = tmp
    pd.DataFrame.from_dict(suites_by_clus, orient = 'index').T.to_csv(output_dir + 
                           '/nos_vs_skillsclusters/suites_organised_by_third_skills_cluster2_v3.csv')
    
    #%%
    '''
    # get list of names for all levels
    all_cluster_names = {'first': list(tax_first_to_second.keys()),
                        'second': list(tax_second_to_first.keys()),
                        'third': list(tax_third_to_second.keys())}
    
    #%
    ''
    #-Most common EVERYTHING per skills cluster.
    #-Arrange it into a hierarchical json for the DATA VIZ 1:
    #    -I think the best thing to do would be to recreate the skills taxonomy 
    #        visualisation with top information about NOS rather than jobs (e.g. top 
    #        suites, top SOC codes, top terms found in titles, top occupations, top 
    #        keywords, this kind of stuff). → the first goal is to create the json, 
    #        like the one Jyl shared with me.
    
    ''
    most_common_features = {}
    most_common_features2 = {}
    for level in ['third','second','first']:
        most_common_features[level] = {}
        most_common_features2[level] = {}
        for cluster_name in all_cluster_names[level]:
            most_common_features[level][cluster_name] = {}
            most_common_features2[level][cluster_name] = {}
            for COL in ['supersuite', 'One_suite', 'SOC4','tokens',
                        'Developed By', 
                        'Keywords', 'Occupations']:
                try:
                    most_common_features[level][cluster_name][COL] = top_values_per_cluster(
                            df_nos2,COL,'tax_{}_level'.format(level),cluster_name)
                    most_common_features2[level][cluster_name][COL] = list(top_values_per_cluster(
                            df_nos2,COL,'tax_{}_level'.format(level),cluster_name).index)
                except:
                    print(level,',', cluster_name,',', COL)
                
    #TODO
    new_tax_hierarchy = {
    "name": "all NOS features",
      "top_suites": "top_suites",
      "all_suites": [],
      "top_SOC": "top_SOC",
      "all_SOC": [],
      "top_keywords": "top_keyword",
      "all_keyword": [],
      "top_terms": "top_terms",
      "all_terms": [],
    
      "avgsalary_range": [],
      "topN_titles": [
        "na",
        0
      ],
      "id": 0}
    
    
    #%%
    '''
    '''
    -SOC vs skills clusters?
    
    cluster_by_soc = {}
    #counter
    for k in most_common_features['second'].keys():
        socs = most_common_features['second'][k]['SOC3']
        cluster_by_soc[k] = {}
        for isoc in socs.index:
            cluster_by_soc[k][isoc] = socs.loc[isoc]
    cluster_by_soc_df = pd.DataFrame.from_dict(cluster_by_soc,orient = 'index')
    plt.figure(figsize = (8,8))
    sns.heatmap(cluster_by_soc_df)
    plt.xlabel('SOC3')
    #plt.savefig(output_dir + '/nos_vs_skillsclusters/skills_clusters_vs_soc3_v3.png',
    #            bbox_inches='tight')
    #plt.close('all')
    
    #%
    
    -Keywords vs skills clusters?
    
    cluster_by_kw = {}
    #counter
    for k in most_common_features['third'].keys():
        socs = most_common_features['third'][k]['Keywords']
        cluster_by_kw[k] = {}
        for isoc in socs.index:
            cluster_by_kw[k][isoc] = socs.loc[isoc]
    cluster_by_kw_df = pd.DataFrame.from_dict(cluster_by_kw,orient = 'index')
    plt.figure(figsize = (8,8))
    sns.heatmap(cluster_by_kw_df)
    plt.xlabel('Keywords')
    #plt.savefig(output_dir + '/nos_vs_skillsclusters/skills_clusters_vs_keywords_v3.png',
    #            bbox_inches='tight')
    #plt.close('all')
    
    
    -Occupations vs skills clusters?
    
    cluster_by_occ = {}
    #counter
    for k in most_common_features['third'].keys():
        socs = most_common_features['third'][k]['Occupations']
        cluster_by_occ[k] = {}
        for isoc in socs.index:
            cluster_by_occ[k][isoc] = socs.loc[isoc]
    cluster_by_occ_df = pd.DataFrame.from_dict(cluster_by_occ,orient = 'index')
    plt.figure(figsize = (8,8))
    sns.heatmap(cluster_by_occ_df)
    plt.xlabel('Occupations')
    #plt.savefig(output_dir + '/nos_vs_skillsclusters/skills_clusters_vs_occ_v3.png',
    #            bbox_inches='tight')
    #plt.close('all')
    
    
    -Which terms from NOS titles are more associated with the various skills clusters?
    
    cluster_by_tokens = {}
    counter
    for k in most_common_features['third'].keys():
        socs = most_common_features['third'][k]['tokens']
        cluster_by_tokens[k] = {}
        for isoc in socs.index:
            cluster_by_tokens[k][isoc] = socs.loc[isoc]
    cluster_by_tokens_df = pd.DataFrame.from_dict(cluster_by_tokens,orient = 'index')
    plt.figure(figsize = (8,8))
    sns.heatmap(cluster_by_tokens_df)
    plt.xlabel('Tokens')
    #plt.savefig(output_dir + '/nos_vs_skillsclusters/skills_clusters_vs_tokens_v3.png',
    #         bbox_inches='tight')
    #plt.close('all')
    
    #%%
    
    -ANYTHING ELSE?
    '''
    
    
    #%%
    '''
    -DATA VIZ2 (?)
        -Similarly, I might want a visualisation by suites (arranged by similarity) 
            and when hovering over a suite, it shows the best cluster. → trial this. 
            Need a json?
    
    
    '''



'''
#%
supersuites = ['engineering', 'management', 'construction', 'financialservices', 'other']
empty = Counter()
# checkpoint
LOAD_DATA = True
if LOAD_DATA:
    print('Loading the data')
    for ix,supersuite in enumerate(supersuites):
        print('Processing supersuite ', supersuite)
        df_nos_select = pd.read_csv(output_dir + 
                             '/augmented_info_NOS_in_supersuites_{}.csv'.format(supersuite))
        df_nos_select.set_index('Unnamed: 0', inplace = True)
        # need to evaluate some columns
        for col in ['SOC4','SOC3','SOC2','SOC1','pruned_lemmas']:
            #print('Processing',col)
            df_nos_select[col]= df_nos_select[col].map(literal_eval)
        for col in ['Salary-peak']:
            #print('Processing',col)
            flag = df_nos_select[col].map(lambda x: isinstance(x,str) & (x!='empty'))
            df_nos_select[col][flag] = df_nos_select[col][flag].map(literal_eval)
        for col in ['tax_third_level']:
            #print('Processing',col)
            flag = df_nos_select[col].map(lambda x: (x[0]=='[') & (x!='empty'))
            df_nos_select[col][flag] = df_nos_select[col][flag].map(literal_eval)
        for col in ['myExp','myEdu','title_processed','converted_skills','London']:
            #print('Processing',col)
            df_nos_select[col]= df_nos_select[col].map(lambda x: eval(x))
        for col in ['Salary']:
            #print('Processing',col)
            df_nos_select[col] =df_nos_select[col].map(lambda x: re.findall(r'[0-9]+',x)).map(
                    lambda x: [float(t) for t in x])
        if ix == 0:
            df_nos = df_nos_select
        else:
            df_nos = df_nos.append(df_nos_select)
'''
            

#%%
'''
Check whether I'm counting twice a m-gram contained in a longer n-gram, where n>m
def check_bigrams(list_skills):
    already_there = []
    for t in list_skills:
        checks = [(t in item) and (t!=item) for item in list_skills]
        if any(checks):
            already_there.append((t, [item for ix,item in enumerate(list_skills) if checks[ix]]))      
    return already_there

def correct_exact_matches(row_nos):
    text_nos = row_nos['clean_full_text']
    exact_matches = row_nos['exact_matches']
    duplications = row_nos['duplications']
    if len(duplications)==0:
        return exact_matches
    skills_to_remove = []
    for short_skill, long_skills in duplications:
        # t is a tuple: (short skill, long skill).
        # I want to check if 'short skill' appears alone without 'long skill'
        # Step 1. remove long skill
        for long_skill in long_skills:
            text_nos_reduced = text_nos.replace(long_skill, '')
        if not short_skill in text_nos_reduced:
            # if now I can't find the short skill, remove it
            skills_to_remove.append(short_skill)
    corrected_exact_matches = [t for t in exact_matches if t not in skills_to_remove]
    return corrected_exact_matches

def subtract_exact_matches(row_nos):
    tmp = [t for t in row_nos['exact_matches'] if t not in row_nos['correct_exact_matches']]
    return tmp


small_df = df_nos2[['exact_matches','clean_full_text']]
small_df['duplications'] = small_df.exact_matches.map(check_bigrams)
small_df['correct_exact_matches'] = small_df.apply(correct_exact_matches, axis =1)
small_df['differences'] = small_df.apply(subtract_exact_matches, axis = 1)
small_small_df = small_df.sample(n=1000)
single_row = small_small_df.iloc[72]
'''