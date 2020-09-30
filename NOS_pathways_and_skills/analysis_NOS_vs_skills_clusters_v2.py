#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 17:14:10 2019

@author: stefgarasto
"""

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
    def add_fl_colours_from_skills(entity_count,first_level_colours,tax_third_to_first,skills_matches_consensus4):
        clrs = list(entity_count.index.map(lambda x: first_level_colours[tax_third_to_first
                                    [skills_matches_consensus4.loc[x]['consensus_cluster']]]))    
        patches = []
        for clr in first_level_colours.keys():
            if clr in ['Uncertain','uncertain']: #clusters_to_exclude:
                continue
            patches.append(mpatches.Patch(color=first_level_colours[clr], label=clr.capitalize()))
        return patches, clrs
    
    def add_fl_colours_from_clusters(entity_count,first_level_colours,tax_third_to_first):
        clrs = list(entity_count.index.map(lambda x: first_level_colours[tax_third_to_first[x]]))    
        patches = []
        for clr in first_level_colours.keys():
            if clr in ['Uncertain','uncertain']: #clusters_to_exclude:
                continue
            patches.append(mpatches.Patch(color=first_level_colours[clr], label=clr.capitalize()))
        return patches, clrs

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

    #%%
    def plot_distribution_from_df(df_nos, plotting_col, first_level_colours,tax_third_to_first,
                                  w=7, h=14, Nplot=50, x_label= 'Number of occurrences',
                                  y_label= 'Skill cluster', 
                                  title= 'Top extracted skill clusters',
                                  save_dir = output_dir, SAVEFIG = False,
                                  save_fig_name = 'my_figure.png'):
        
        try:
            cluster_series = pd.DataFrame(flatten_lol(df_nos[plotting_col].values))
            cluster_count = cluster_series[0].value_counts()
        except:
            cluster_count = df_nos
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
     
    #%%
    def autolabel(bars,ax,add_value = 1.1):
        # attach some text labels
        widths = []
        for bar in bars:
            widths.append(bar.get_width())
        a_max = 0.02 if np.median(widths)<0.4 else 1
        for bar in bars:
            width0 = bar.get_width()
            if np.abs(width0)<1e-6:
                # check if it's 0
                width0 = 1
                wid_label = '0'
            elif np.abs(width0)<=0.05:
                # check if the approximation would give 0
                width0 = 1
                wid_label = '<0.1'
            else:
                # approximate to first decimal digit
                width0 = np.around(bar.get_width(),1)
                wid_label = '{}'.format(width0)
            ax.text(np.max([width0+add_value,a_max]), bar.get_y() + bar.get_height()/2-.03,
                    wid_label, ha='right', va='center',fontsize = 10.5)
            
    
                
    #%%            
    def plot_two_distribution_from_df(df_nos, plotting_col, comparison_dist,
                                      first_level_colours,tax_third_to_first,
                                      w=7, h=14, Nstart= 0, Nplot=50, x_label= 'Number of occurrences',
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
        if Nstart is None:
            Nstart = 0
            
        ind = np.arange(Nplot-Nstart-1,-1,-1)
        #print(ind)
        width = 0.375
        
        # make the plots
        a_vals = a_vals/a_vals.sum()*100
        #a_vals = a_vals[::-1]
        #b_vals = b_vals[::-1]
        #print(a_vals.values[:10],b_vals.values[:10])
        a = ax.barh(ind+ width, a_vals.values[Nstart:Nplot], width, color = 
                    clrs[Nstart:Nplot], linestyle = '-', edgecolor ='k',
                    linewidth=.8) # plot a vals
        b = ax.barh(ind, b_vals.values[Nstart:Nplot], width*.9, color = 
                    clrs[Nstart:Nplot], alpha=0.7, hatch = '//',
                    edgecolor = 'k', linewidth=1.2)
                    #linestyle = '--')

#                    clrs[Nstart:Nplot], linewidth=2)  # plot b vals
        
        plt.ylim([-.5,Nplot-Nstart])
        ax.set_yticks(ind + width)  # position axis ticks
        ax.set_yticklabels(a_vals.index[Nstart:Nplot])  # set them to the names
        
        patches.append(mpatches.Patch(fc=None, linestyle = '-',
                                      linewidth = 2, fill=False,
                                      ec = 'black',
                                      label='Proportion of NOS'))
        patches.append(mpatches.Patch(fc=None, fill=False,
                                      ec = 'black',
                                      linestyle = '-', #'--'
                                      hatch = '//',
                                      linewidth=2, label='Proportion of job adverts'))
        
        ax.legend(handles = patches, # + [a[0], b[0]], ['Proportion of NOS', 'Proportion of jobs'], 
                  loc='lower right')
        
        if plotting_col == 'assigned_third':
            if Nstart == 0:
                add_value = 0.4
            elif (Nstart >30) & (Nstart<60):
                add_value = 0.01
            else:
                add_value = 0.0
        else:
            add_value = 1.1
        autolabel(a,ax,add_value)
        #add_text_to_hist_new(values, xvalues = None, addval = None, orient = 'vertical')
        autolabel(b,ax,add_value)
        
        plt.xlabel(x_label, fontsize = 18)
        plt.ylabel(y_label, fontsize = 18)
        plt.title(title, fontsize =20)
        T = plt.yticks()
        T = capitalise_labels(T)
        _ = plt.yticks(T[0],T[1])
        plt.tight_layout()
        if SAVEFIG:
            plt.savefig(os.path.join(save_dir, save_fig_name))
            
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

    #%
    from extract_skills_from_nos import exact_skills_from_standards, fuzzy_skills_from_standards, \
        skills_matches_consensus2, skills_matches_consensus4, skills_lemmas

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
    
    
#%% checkpoint
EXAMINE_SKILLS = True
SAVEFIG = True
if EXAMINE_SKILLS:
    #%% how many skills per NOS?
    skills_nb_count = df_nos2[skills_col2use].map(len)#.value_counts()
    f = plt.figure(figsize = (7,5))
    sns.distplot(skills_nb_count, color= nesta_colours[3], kde = False, norm_hist = False)
    avg_skill_nb = np.mean(skills_nb_count)
    plt.plot([avg_skill_nb,avg_skill_nb],[0,2900],'--',color=nesta_colours[3])
    plt.text(avg_skill_nb + 2, 2850, '{}'.format(np.around(avg_skill_nb,2)),
             fontdict={'size': 14})
    plt.ylabel('Number of NOS', fontsize = 18)
    plt.xlabel('Number of skills extracted', fontsize = 18)
    plt.tight_layout()
    if SAVEFIG:
        plt.savefig(os.path.join(matches_dir, 'number_of_skills_per_NOS_v3.png'))
    
    #%%
    ''' which skills were found more frequently?'''
    skills_series = pd.DataFrame(flatten_lol(df_nos2[skills_col2use].values))
    # add first level clusters
    skills_count = skills_series[0].value_counts()
    print('{} out of {} skills were found'.format(len(skills_count),
          len(skills_matches_consensus4)))
    
    ''' which are the most mentioned skills?'''
    f = plt.figure(figsize = (9,12))
    clrs = list(skills_count.index.map(lambda x: first_level_colours[tax_third_to_first
                                [skills_matches_consensus4.loc[x]['consensus_cluster']]]))    

    sns.countplot(y = 0, data=skills_series, order = skills_count.index[:50], palette = clrs[:50])
    patches = []
    for clr in First_level_colours.keys():
        if clr in ['Uncertain','uncertain']: #clusters_to_exclude:
            continue
        patches.append(mpatches.Patch(color=First_level_colours[clr], label=clr))
    plt.legend(handles=patches, title = 'FL skill cluster', title_fontsize = 17)
    add_text_to_hist_new(skills_count[:50].values, xvalues= np.arange(.2,50.2))
    plt.xlabel('Number of occurrences', fontsize = 18)
    plt.ylabel('Skill', fontsize = 18)
    plt.title('Top extracted skills', fontsize = 20)
    T = plt.yticks()
    T = capitalise_labels(T)
    _ = plt.yticks(T[0],T[1])
    plt.tight_layout()
    if SAVEFIG:
        plt.savefig(os.path.join(matches_dir, 'most_common_skills_extracted_v3.svg'))
    
    #%%
    '''
    Skills found by suite: most common skills and most common clusters
    mentioned
    
    '''
    suite_groups = df_nos2.groupby('One_suite')
    extract_skills_membership = {}
    for name, g in suite_groups:
        # take the top 20 skills for each cluster
        if len(g)<5:
            continue
        top10skills = [t[0].capitalize()
            for t in Counter(flatten_lol(g[skills_col2use].values)
            ).most_common()[:20]]
        top20clusters = [t[0].capitalize() for t in Counter(flatten_lol(g['exact_third_level'])).most_common()[:20]
                            if t not in ['Uncertain','uncertain']] #clusters_to_exclude]
                #g['exact_cluster'].values)
        extract_skills_membership[name.capitalize()] = {'Top skills extracted':'; '.join(top10skills),
                                 'Corresponding top clusters': '; '.join(top20clusters),
                                 'Suite size': len(g)}
    skills_compare_by_suite = pd.DataFrame.from_dict(extract_skills_membership,
                                     orient ='index')
    skills_compare_by_suite = skills_compare_by_suite.sort_values('Suite size',
                                                                  ascending=False)
    if SAVEFIG:
        skills_compare_by_suite.to_csv(os.path.join(matches_dir,'skills_extracted_by_suite_v3.csv'))
    
                 
    
    #%%
    ''' Appendix 1'''
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
    '''Appendix 2. Which third level clusters do extracted skills come from?'''
    Nplot = 50
    cluster_series = pd.DataFrame(flatten_lol(df_nos2['exact_third_level'].values))
    tl_extracted = cluster_series[0].value_counts()
    f = plt.figure(figsize = (11,12))
    patches, clrs = add_fl_colours_from_clusters(tl_extracted,first_level_colours,tax_third_to_first)
    
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
    if SAVEFIG:
        plt.savefig(os.path.join(matches_dir, 'most_common_clusters_extracted_third_level_v3.png'))
    
    #%
    print('The less common third level clusters are:')
    print(tl_extracted[:50])
    
    print('TL clusters that are never mentioned are: ')
    print(set(tax_third_to_second.keys()) - set(tl_extracted.index))
    
    
    #%
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
    
    #% as an example, print the missing skills for driving and welding
    example_missing = missing_clusters[(missing_clusters.map(lambda x: x in 
            ['welding and machining','driving and automotive maintenance']))]
    example_missing= pd.DataFrame(example_missing.sort_values()).rename(columns = {'consensus_cluster':'Skill cluster'})
    example_missing.index.rename('Skill name',inplace=True)
    if SAVEFIG:
        pd.DataFrame(example_missing).to_csv(os.path.join(matches_dir,
                    'missing_skills_driving_and_welding_v3.csv'))
    
    
    #%%    
    ''' Benchmarking against taxonomy'''
    ''' Plot distribution of TL clusters coloured by FL'''
    plot_distribution_from_df(df_nos2,'assigned_third', first_level_colours, tax_third_to_first,
                                  w=10.5, h=26, Nplot=None, x_label= 'Number of occurrences',
                                  y_label= 'Skill cluster', 
                                  title= 'Top extracted skill clusters',
                                  save_dir = matches_dir + '/nos_vs_skillsclusters', 
                                  SAVEFIG = False,
                                  save_fig_name = 'NOS_skills_clusters_for__third_level_title_v3.png')
    
    #%%
    ''' Also plot with comparison to job adverts, in case I need'''
    indices_start = [0,50,100]
    indices_end = [50,100,None]
    heights = [14,14,10]
    comparison_dist = pd.DataFrame.from_dict(prop_jobs_dict, orient='index',
                                           columns =['Proportion of jobs'])
    for ix in range(3):
        Nstart = indices_start[ix]
        Nplot = indices_end[ix]
        plot_two_distribution_from_df(df_nos2, 'assigned_third', 
                                comparison_dist['Proportion of jobs'],
                                first_level_colours,tax_third_to_first,
                                w=10.5, h=heights[ix], Nstart=Nstart, Nplot=Nplot, 
                                x_label= 'Proportions (%)',
                                y_label= 'Skill cluster', 
                                title= 'Assigned skill clusters (part {})'.format(ix+1),
                                save_dir = matches_dir + '/nos_vs_skillsclusters', 
                                SAVEFIG = True,
                                save_fig_name = 'NOS_skills_clusters_for_third_level_title_compare_{}_v4.svg'.format(ix))
    
    #%% plot distributions side by side on SLSC

    comparison_dist['SL'] = comparison_dist.index.map(lambda x: tax_third_to_second[x]
            if x in tax_third_to_second.keys() else None)
    comparison_dist_sl = comparison_dist.groupby('SL').agg(sum)
    plot_two_distribution_from_df(df_nos2, 'assigned_second', 
                                comparison_dist_sl['Proportion of jobs'],
                                first_level_colours,tax_second_to_first,
                                w=11, h=12, Nstart=0, Nplot=None, 
                                x_label= 'Proportions (%)',
                                y_label= 'SL skill cluster', 
                                title= '',
                                save_dir = matches_dir + '/nos_vs_skillsclusters', 
                                SAVEFIG = True,
                                save_fig_name = 'NOS_skills_clusters_for_second_level_title_compare.svg')
    
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
    '''
    Collect all relevant stuff in dataframe
    '''
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
    
    # remove clusters with virtually no advrets
    numeric_df = numeric_df[numeric_df.index.map(lambda x: x not in clusters_to_exclude)]
    
    #%% compute ratio
    numeric_df['ratio'] = numeric_df['Proportion of NOS']/numeric_df['Proportion of jobs']
    numeric_df.ratio[numeric_df.ratio==0] = np.nan
    # discard clusters with no actual representation in job adverts
    numeric_df['log_ratio'] = numeric_df.ratio.map(np.log)
    
    #%%
    # scatter plot, growth vs representation
    plt.figure(figsize=(10,6))
    col2 = 'log_ratio' #'Proportion of NOS'
    fig = sns.scatterplot(x = 'Growth', y = col2, data = numeric_df, 
                     hue = 'FL skill cluster', palette = First_level_colours,
                     alpha = .7)
    fig.legend(loc="upper left")#, bbox_to_anchor=(1.05, 1))
    plt.xlabel('Skill cluster growth',fontsize = 18)
    plt.ylabel('Representation factor',fontsize = 18)
    plt.ylim([-4.8,4.8])
    plt.tight_layout()
    # high growth clusters and the top 25% clusters in terms of growth (top quartile)
    # and with more than 0.01% share of the market
    high_growth_th = np.percentile(numeric_df['Growth'],75)
    high_growth = numeric_df[(numeric_df['Growth']>high_growth_th) &
                             (numeric_df['Proportion of jobs']>0.01) &
                             (numeric_df['log_ratio']<.6)]
    growths_list = high_growth.index
    high_growth = high_growth.sort_values('Growth',ascending= False)
    for name,t in high_growth[:10].iterrows():
        #print(name)
        plt.scatter(t['Growth'],t[col2],s=80,facecolors ='none',edgecolors='k')
        if 'assistance and care' in name:
            plt.text(t['Growth']+.02,t[col2]-.15,name.capitalize(),fontsize = 11)
        elif 'engineering' in name:
            plt.text(t['Growth']-.2,t[col2]+.2,name.capitalize(),fontsize = 11)
        elif 'dental assistance' in name:
            plt.text(t['Growth']+.02,t[col2]-.04,name.capitalize(),fontsize = 11)
        elif name in ['phlebotomy','data engineering','physiotherapy and beauty']:
            plt.text(t['Growth']+.02,t[col2]+.1,name.capitalize(),fontsize = 11)
        else:
            plt.text(t['Growth']+.02,t[col2],name.capitalize(),fontsize = 11)
    for name in ['web development', 'software development','design and process engineering']:
        t = numeric_df.loc[name]
        plt.scatter(t['Growth'],t[col2],s=80,
                    facecolors =First_level_colours[t['FL skill cluster']],edgecolors='k')
        plt.text(t['Growth']-.25,t[col2]-.25,name.capitalize(),fontsize = 11)
    
    name= 'app development'
    t = numeric_df.loc[name]
    plt.plot([t['Growth'],t['Growth']],[-4.8,4.8],'--',color=nesta_colours[5])
    plt.text(t['Growth']+.02,4.4,'App development',color = nesta_colours[5], 
             fontsize =14)

    #name= 'web development'
    #t = numeric_df.loc[name]
    #plt.scatter(t['Growth'],t[col2],s=80,facecolors ='none',edgecolors='k')
    #plt.text(t['Growth']-.2,t[col2]-.25,name.capitalize(),fontsize = 11)
    
    xticks_list = [-35,-10,-3,1,3,10,35] #[-4,-2,0,2,4]
    plt.yticks([np.log(-1/t) if t<0 else np.log(t) for t in xticks_list],
                    ['{}x'.format(t) for t in xticks_list])
    
    if SAVEFIG:
        plt.savefig(output_dir + '/nos_vs_skillsclusters/NOS_skills_clusters_vs_growth_v4.png')
    
    #%% scatter plot, median salary vs representation. Third level clusters.
    col = 'median_salary'
    col2 = 'log_ratio' #'Proportion of NOS'
    plt.figure(figsize=(10,6))
    fig = sns.scatterplot(x = col, y = col2, data = numeric_df, 
                     hue = 'FL skill cluster', palette = First_level_colours,
                     alpha = .7)
    fig.legend(loc="upper left")#, bbox_to_anchor=(1.05, 1))
    plt.xlabel('Skill cluster median salary',fontsize = 18)
    plt.ylabel('Representation factor',fontsize = 18)
    plt.ylim([-4.8,4.8])
    plt.tight_layout()
    # high growth clusters and the top 25% clusters in terms of growth (top quartile)
    # and with more than 0.01% share of the market
    high_growth_th = np.percentile(numeric_df[col][~numeric_df[col].isnull()],75)
    high_growth = numeric_df[(numeric_df[col]>high_growth_th) &
                             (numeric_df['Proportion of jobs']>0.01) &
                             (numeric_df['log_ratio']<0)]
    salaries_list = high_growth.index
    high_growth = high_growth.sort_values(col,ascending= False)
    for name,t in high_growth[:10].iterrows():
        plt.scatter(t[col],t[col2],s=80,facecolors ='none',edgecolors='k')
        plt.text(t[col]+200,t[col2]-.25,name.capitalize(),fontsize = 11)
    for name in ['web development', 'software development','design and process engineering']:
        t = numeric_df.loc[name]
        plt.scatter(t[col],t[col2],s=80,
                    facecolors =First_level_colours[t['FL skill cluster']],edgecolors='k')
        plt.text(t[col]-4500,t[col2]-.35,name.capitalize(),fontsize = 11)
    
    name= 'app development'
    t = numeric_df.loc[name]
    plt.plot([t[col],t[col]],[-4.8,4.8],'--',color=nesta_colours[5])
    plt.text(t[col]+100,4.4,'App development',color = nesta_colours[5], 
             fontsize =14)
    #plt.scatter(t[col],t[col2],s=80,facecolors ='none',edgecolors='k')
    #plt.text(t[col]-.2,t[col2]-.25,name.capitalize(),fontsize = 11)

    xticks_list = [-35,-10,-3,1,3,10,35] #[-4,-2,0,2,4]
    plt.yticks([np.log(-1/t) if t<0 else np.log(t) for t in xticks_list],
                    ['{}x'.format(t) for t in xticks_list])
    
    if SAVEFIG:
        plt.savefig(output_dir + '/nos_vs_skillsclusters/NOS_skills_clusters_vs_salary_v4.svg')
    
    
    #%%
    A = numeric_df[['ratio','Proportion of jobs','Proportion of NOS','Growth','log_ratio']
    ].sort_values('ratio')

    list_for_labels = list(set(salaries_list).union(growths_list))+ [
        'design and process engineering','app development','aviation','construction',
        'welding and machining','hr management','it security standards','animation',
        'business management','physiotherapy and beauty','logistics administration']
    # discard clusters with no actual representation in job adverts
    notna = A.ratio.map(lambda x: not np.isnan(x))

    heights = [0,16,18]
    ih =17
    job_th = 0.01
    for jx in [-1,1]:
        if jx==1:
            smallA = A[['ratio','log_ratio']][(A.log_ratio>=np.log(1)) & 
                  (A['Proportion of jobs']>job_th) & 
                         (A['Proportion of NOS']>0)]
        else:
            smallA = A[['ratio','log_ratio']][(A.log_ratio<=np.log(1)) & 
                  (A['Proportion of jobs']>job_th) & 
                         (A['Proportion of NOS']>0)]

        # 0.1% means roughly 50,000 job adverts
        # 0.001% means roughly 500 job adverts
        f = plt.figure(figsize = (10,ih))
        patches, clrs = add_fl_colours_from_clusters(smallA['log_ratio'],first_level_colours,
                                                         tax_third_to_first)
        smallA['log_ratio'].plot('barh', color= clrs) #nesta_colours[3])
        plt.xlim([-5,5])
        plt.ylim([-2,len(smallA)])
        # add label with the proper ratio
        for ix,i in enumerate(smallA['log_ratio']):
            if smallA.index[ix] not in list_for_labels:
                continue
            x = ix - .2
            if i>0:
                ii = np.around(smallA.iloc[ix]['ratio'],1)
                plt.text(i+.1, x, '+{}x'.format(ii), fontsize = 12)
            else:
                ii = np.around(1/smallA.iloc[ix]['ratio'],1)
                plt.text(i-.9, x, '-{}x'.format(ii), fontsize = 12)
        
        if ix==-1:
            plt.xlabel(''.join(['(Under-)representation ',
                            'factor']), fontsize = 17)
        else:
            plt.xlabel(''.join(['(Over-)representation ',
                            'factor']), fontsize = 17)
        #plt.xlabel(''.join(['Over-representation (positive numbers) \n ',
        #                    'and under-representation (negative numbers) \n ',
        #                    'factor']), fontsize = 17)
        plt.ylabel('Skill cluster', fontsize = 17)
        T = plt.yticks()
        T = capitalise_labels(T)
        _ = plt.yticks(T[0],T[1],verticalalignment='baseline')
        xticks_list = [-35,-10,-3,1,3,10,35] #[-4,-2,0,2,4]
        for t in xticks_list:
            if t*jx<0:
                continue
            xt = np.log(-1/t) if t<0 else np.log(t)
            plt.plot([xt,xt],[-2,len(smallA)],'k--',linewidth=1)
        #plt.xticks(xticks_list,[np.around(np.exp(t)) for t in xticks_list])
        plt.xticks([np.log(-1/t) if t<0 else np.log(t) for t in xticks_list],
                    ['{}x'.format(t) for t in xticks_list])
        plt.legend(handles=patches, loc='lower right', bbox_to_anchor=(1.05, 1),
                   title = 'FL skill cluster', title_fontsize = 17)
        plt.tight_layout()
        if SAVEFIG:
            plt.savefig(output_dir + 
                '/nos_vs_skillsclusters/NOS_vs_jobs_representation_factor_{}_{}_v4.svg'.format(
                        job_th,ix))
      
    #%% replot representation ratio at the SL skill cluster
    list_for_labels_sl = ['software engineering', 'it systems and support',
       'windows programming',
       'business intelligence and it systems design', 'dentistry',
       'design',
       'energy and environmental management',
       'construction, maintenance and transport', 'management and hr']
    numeric_df['SL skill cluster'] = numeric_df.index.map(lambda x: tax_third_to_second[x])
    numeric_df_sl = numeric_df.groupby('SL skill cluster').agg({'Growth': np.nanmean, 
            'median_salary': np.nanmean, 'Proportion of jobs': sum, 
            'Proportion of NOS': sum, 'Proportion of skills': sum})
    numeric_df_sl['FL skill cluster'] = numeric_df_sl.index.map(lambda x: tax_second_to_first[x])
    numeric_df_sl['ratio']= numeric_df_sl['Proportion of NOS']/numeric_df_sl['Proportion of jobs']
    numeric_df_sl.ratio[numeric_df_sl.ratio==0] = np.nan
    # discard clusters with no actual representation in job adverts
    numeric_df_sl['log_ratio'] = numeric_df_sl.ratio.map(np.log)
    numeric_df_sl = numeric_df_sl.sort_values('log_ratio')
    notna = numeric_df_sl.ratio.map(lambda x: not np.isnan(x))

    heights = [0,16,18]
    ih =12
    job_th = 0.01
    smallA = numeric_df_sl[['ratio','log_ratio']][(numeric_df_sl['Proportion of jobs']>job_th) & 
                         (numeric_df_sl['Proportion of NOS']>0)]

    # 0.1% means roughly 50,000 job adverts
    # 0.001% means roughly 500 job adverts
    f = plt.figure(figsize = (10,ih))
    patches, clrs = add_fl_colours_from_clusters(smallA['log_ratio'],first_level_colours,
                                                     tax_second_to_first)
    smallA['log_ratio'].plot('barh', color= clrs) #nesta_colours[3])
    plt.xlim([-3,3])
    plt.ylim([-2,len(smallA)])
    # add label with the proper ratio
    for ix,i in enumerate(smallA['log_ratio']):
        #if smallA.index[ix] not in list_for_labels_sl:
        #    continue
        x = ix - .2
        if i>0:
            ii = np.around(smallA.iloc[ix]['ratio'],1)
            plt.text(i+.1, x, '+{}'.format(ii), fontsize = 12)
        else:
            ii = np.around(1/smallA.iloc[ix]['ratio'],1)
            plt.text(i-.5, x, '-{}'.format(ii), fontsize = 12)
    
    plt.xlabel(''.join(['Representation ',
                        'factor']), fontsize = 17)
    plt.ylabel('Skill cluster', fontsize = 17)
    T = plt.yticks()
    T = capitalise_labels(T)
    _ = plt.yticks(T[0],T[1],verticalalignment='baseline')
    xticks_list = [-10,-5,-2,1,2,5,10] #[-4,-2,0,2,4]
    for t in xticks_list:
        xt = np.log(-1/t) if t<0 else np.log(t)
        plt.plot([xt,xt],[-2,len(smallA)],'k--',linewidth=1)
    #plt.xticks(xticks_list,[np.around(np.exp(t)) for t in xticks_list])
    plt.xticks([np.log(-1/t) if t<0 else np.log(t) for t in xticks_list],
                ['{}'.format(t) for t in xticks_list])
    plt.legend(handles=patches, loc='lower right', bbox_to_anchor=(1.05, 1),
               title = 'FL skill cluster', title_fontsize = 17)
    plt.tight_layout()
    if SAVEFIG:
        plt.savefig(output_dir + 
            '/nos_vs_skillsclusters/NOS_vs_jobs_representation_factor_SL_v4.svg')
    
    #%% scatter plot, median salary vs representation - SL skill cluster
    high_growth_th = np.percentile(numeric_df_sl['Growth'][~numeric_df_sl['Growth'].isnull()],75)
    high_sal_th = np.percentile(numeric_df_sl['median_salary'][~numeric_df_sl['median_salary'].isnull()],75)
    high_growth = numeric_df_sl[(numeric_df_sl['Growth']>high_growth_th) | (numeric_df_sl['median_salary']>high_sal_th)]
    high_growth = high_growth[(high_growth['Proportion of jobs']>0.1) &
                                 (high_growth['log_ratio']<1000)]
    for col in ['median_salary','Growth']:
        #col = 'median_salary'
        col2 = 'log_ratio' #'Proportion of NOS'
        plt.figure(figsize=(10,6))
        fig = sns.scatterplot(x = col, y = col2, data = numeric_df_sl, 
                         hue = 'FL skill cluster', palette = first_level_colours,
                         alpha = .7)
        fig.legend(loc="upper left")#, bbox_to_anchor=(1.05, 1))
        plt.xlabel('Skill cluster {}'.format(col.replace('_',' ').lower()),fontsize = 18)
        plt.ylabel('Representation factor',fontsize = 18)
        plt.ylim([-2.5,2])
        xticks_list = [-10,-5,-2,1,2,5]
        
        plt.yticks([np.log(-1/t) if t<0 else np.log(t) for t in xticks_list],
                    ['{}'.format(t) for t in xticks_list])
        if col == 'median_salary':
            plt.plot([high_sal_th,high_sal_th],[-2.5,2],'--',color = nesta_colours[5])
            plt.text(high_sal_th+200, 1.8,'Top 25% earnings',fontsize =13, 
                     color = nesta_colours[5])
        # high growth clusters and the top 25% clusters in terms of growth (top quartile)
        # and with more than 0.01% share of the market
        plt.tight_layout()
        high_growth = high_growth.sort_values(col,ascending= False)
        for name,t in high_growth[:15].iterrows():
            plt.scatter(t[col],t[col2],s=80,facecolors ='none',edgecolors='k')
            if col == 'median_salary':
                plt.text(t[col]+200,t[col2]-.15,name.capitalize(),fontsize = 11)
            else:
                plt.text(t[col]+.005,t[col2]+.1,name.capitalize(),fontsize = 11)
        
        if SAVEFIG:
            plt.savefig(output_dir + '/nos_vs_skillsclusters/NOS_skills_clusters_vs_{}_SL_v4.svg'.format(col))
            
    #%% 
    print('-'*70)
    print('Comparison ratio for high growth clusters.')
    print('(nan = 0 NOS)')
    print()
    for name,t in high_growth.iterrows():
        print(name, (numeric_df.loc[name]['ratio']))
    
    print('-'*30)
    print('Growth for highly underrepresented clusters:')
    print()
    for name in numeric_df[numeric_df['ratio']<=1/5].sort_values('ratio').index:
        print(name,numeric_df.loc[name]['Growth'])
    
#%%
stop
#%%
'''
To save the outputs
'''
# skills per NOS
A = df_nos2.reset_index()[['index','URN','NOS Title','correct_exact_matches']]
A['URN'] = A['URN'].map(lambda x: x.upper() if isinstance(x,str) else [t.upper() for t in x])
A['correct_exact_matches'] = A['correct_exact_matches'].map(lambda x: '; '.join([t.capitalize() for t in x]))
A['NOS Title'] = A['NOS Title'].map(lambda x: x.strip().capitalize())
A.to_csv(os.path.join(matches_dir,'list_of_skills_per_NOS.csv'))

# skills frequency
B= skills_count
B.index= B.index.map(lambda x: x.capitalize())
B.to_csv(os.path.join(matches_dir,'frequency_of_skills_extracted_from_NOS.csv'))

# dist of NOS by SC
numeric_df_sl['representation_factor'] = numeric_df_sl.ratio.map(lambda x: np.exp(np.abs(np.log(x)))*np.sign(np.log(x)))

A2 = numeric_df[['Proportion of NOS','representation_factor']]
A2 = A2.join(pd.DataFrame(pd.Series(flatten_lol(
            df_nos2['assigned_third'])).value_counts()))
Nc2 = np.nansum(A2[0])
A2.index = A2.index.map(lambda x: x.capitalize())
A2.to_csv(os.path.join(matches_dir,'NOS_by_third_level_clusters.csv'))

B2 = numeric_df_sl[['Proportion of NOS','representation_factor']]
B2['Number of NOS'] = np.round(B2['Proportion of NOS']/100*np.nansum(A2[0])).map(lambda x: int(x))
B2.index = B2.index.map(lambda x: x.capitalize())
B2.to_csv(os.path.join(matches_dir,'NOS_by_second_level_clusters.csv'))
