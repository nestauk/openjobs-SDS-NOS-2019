# coding: utf-8
#!/usr/bin/env python


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
    
    #from gensim.scripts.glove2word2vec import glove2word2vec
    
    
    # In[163]:
    
    
    from utils_bg import * #nesta_colours, nesta_colours_combos
    from utils_nlp import *
    from utils_skills_clusters import load_and_process_clusters
    
    
    from map_NOS_to_pathways_utils import *
    
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
    pofs_titles = 'nj'
    
    paramsn = {}
    paramsn['ngrams'] = 'bi'
    paramsn['pofs'] = pofs
    USE_TITLES = True
    USE_KEYWORDS = False
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
    
    
        #return [df_line['tagged_tokens'][0][:len(df_line['NOS Title'])]]
    # create another column where the texts are lemmatised properly
    if USE_TITLES:
        t0 = time.time()
        df_nos['pruned_lemmas'] = df_nos.apply(extract_tagged_title, axis=1).map(
                lambda x: lemmatise_pruned(x,pofs_titles))#pofs))
        print_elapsed(t0, 'lemmatising tagged tokens using only titles')        
    elif USE_KEYWORDS:
        t0 = time.time()
        df_nos['pruned_lemmas'] = df_nos['Keywords'].map(lambda x: get_keywords_list(x, 
                          stopwords0)).map(lambda x: process_keywords(x,model))#.map(
                #lambda x: lemmatise_pruned(x,pofs_titles))#pofs))
        print_elapsed(t0, 'lemmatising tagged tokens using only expert keywords')        
    else:
        t0 = time.time()
        df_nos['pruned_lemmas'] = df_nos['tagged_tokens'].map(
                lambda x: lemmatise_pruned(x,pofs))
        print_elapsed(t0, 'lemmatising tagged tokens using full text')
    
    
    # ### Only keep NOS from a super-suite
    
    # In[21]:
    
    # load which suites are in each super-suite
    super_suites_files=  ''.join(['/Users/stefgarasto/Google Drive/Documents/data/',
                                  'NOS_meta_data/NOS_Suite_Priority.xlsx'])
    super_suites_names = ['Engineering','Management','FinancialServices','Construction']
    all_super_suites = {}
    for which_super_suite in super_suites_names:
        all_super_suites[which_super_suite] = pd.read_excel(super_suites_files, 
                        sheet_name = which_super_suite)
        all_super_suites[which_super_suite]['NOS Suite name'] = all_super_suites[
            which_super_suite]['NOS Suite name'].map(
            lambda x: x.replace('(','').replace('(','').replace('&','and').strip().lower())
    
    # In[22]:
    
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
    
    
    # In[32]:
    # assign supersuite and SOC codes
    df_nos['supersuite'] = df_nos['One_suite'].apply(assign_supersuite)
    # extract 2 digit soc
    df_nos['SOC4str'] = df_nos['Clean SOC Code'].map(adjustsoccode)
    df_nos['SOC1'] = df_nos['SOC4str'].map(extract1digits)
    df_nos['SOC2'] = df_nos['SOC4str'].map(extract2digits)
    df_nos['SOC3'] = df_nos['SOC4str'].map(extract3digits)
    df_nos['SOC4'] = df_nos['SOC4str'].map(extract4digits)
    print(df_nos['supersuite'].value_counts())
    
    
    # In[38]:
    
    
    # select NOS in super-suites of interest
    ONLY_ENG = True
    if ONLY_ENG:
        df_nos_select = df_nos[df_nos['supersuite']=='engineering']
    else:
        df_nos_select = df_nos[~(df_nos['supersuite']=='other')]
    print(len(df_nos_select))
    
    
    #%%
    '''
    # ## Get raw data and tokenize
    
    # ## Choosing parameters for features extraction
    # 
    # ngrams : uni/bi/tri
    # 
    # tfidf thresholds: min and max percentage
    # 
    # which parts of speech were selected before
    # 
    # whether we are working at the level of suites or of invidual NOS, 
    # and how we aggregate NOS to form the suit level
    # 
    '''
    #
    
    '''
    # First, create your TFidfVectorizer model. This doesn't depend on whether 
    it's used on suites or NOS. However,
    it does require that the docs collection is already given as a collection of
    tokens (tokenizer=tokenize_asis)
    
    #Since we now have not just long strings in our documents, but lists of terms, 
    we will use a different tokenizer
    '''
        
        
    # define the transform: this one can easily be the same for both 
    # keywords and the clustering
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
    
    
    
    # In[44]:
    
    
    SAVEKW= False
        
    # In[47]:
    
    
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
        #top_terms = get_top_words(df_nos_select.loc[name]['pruned'], 
                        #feature_names_n, tfidf, n = 20)
        #top_ngrams = np.argsort(tfidfm_dense[ix,:])
        #top_ngrams = top_ngrams.tolist()[0][-20:]
        #top_ngrams = top_ngrams[::-1]
        ## only retain the ones with non zero features
        #top_ngrams = [elem for elem in top_ngrams if tfidfm_dense[ix,elem]>0]
        #top_weights = [tfidfm_dense[ix,elem] for elem in top_ngrams]
        #top_features = [feature_names_n[elem] for elem in top_ngrams]
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
    if SAVEKW:
        pd.DataFrame.from_dict(top_terms_dict, orient = 'index').to_csv(output_dir +
                                                    '/NOS_from_supersuites_top_terms_{}_{}.csv'.format(qualifier,pofs))
        pd.DataFrame.from_dict(top_weights_dict, orient = 'index').to_csv(output_dir +
                              '/NOS_from_supersuites_top_terms_weights_{}_{}.csv'.format(qualifier,pofs))
    tfidfm_dense = None
    
    
    # In[78]:
    
    
    # just to check results
    '''
    print(list(top_terms_dict.keys())[885:887])
    top_terms_weights = get_top_words_weights([df_nos_select.iloc[0]['pruned_lemmas']], feature_names_n, tfidf, n = 20)
    print(top_terms_weights.sort_values(by = 'tfidf', ascending = False).head(n=20))
    '''
    # note that the get_top_words_weights function is probably wrong - but it doesn't matter now
    print('not now')
    
    
    # In[82]:
    
    
    # remove top terms that are not in the chosen gensim model
    new_top_terms_dict = {}
    new_top_weights_dict = {}
    for k,v in top_terms_dict.items():
        if len(v)==0:
            new_top_terms_dict[k] = v
            continue
        # check if the top terms for each document are in the gensim model
        if (paramsn['ngrams']=='bi') | USE_KEYWORDS:
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
    clus_names,comparison_vecs,skill_cluster_vecs =load_and_process_clusters(model,
                                                                        ENG=True)
    
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

#%%
        '''
        if (len(tmp_clus)!=2) & USE_WEIGHTED:
            counter_clus = Counter(weighted_clus).most_common() #tmp_clus.most_common()
            acceptability_th = 0.01
        else:
            counter_clus = tmp_clus.most_common()
            acceptability_th = 1
        if len(counter_clus)==2:
            if tmp_clus[0][1]==tmp_clus[1][1]:
                # only two clusters have been individuated an equal number of times
                st_v_clus[k] = [t[0] for t in tmp_clus]
            else:
                st_v_clus[k] = tmp_clus[0][0]
        elif tmp_clus[0][1]>1:
            st_v_clus[k] = tmp_clus[0][0]
        else:
            st_v_clus[k] = highest_similarity_threshold_top_two(test_skills, comparison_vecs, 
                     clus_names, th = 0.3) #tmp_bkp
        if not isinstance(st_v_clus[k],list):
            st_v_clus[k] = [st_v_clus[k]]
        '''
        #st_v_clus2[k] = highest_similarity(test_skills, comparison_vecs, clus_names)

        
    
    # In[ ]:
    
    
    # add the best clusters to the nos dataframe
    tmp = pd.DataFrame.from_dict(st_v_clus, orient = 'index')
    tmp = tmp.rename(columns = {0: 'best_cluster_nos'})
    df_nos_select['best_cluster_nos'] = tmp['best_cluster_nos']
    
    
    #%%
    # compare with skills clusters based on full text for selected nos
    examples_file= ''.join(['/Users/stefgarasto/Google Drive/Documents/results/',
                            'NOS/progression_pathways/nos_and_skills_clusters/',
                            'sample_nos_and_skills_clusters1.csv'])
    examples_data = pd.read_csv(examples_file)
    examples_data.set_index('Unnamed: 0',inplace=True)
    examples_data['best_cluster_nos_title'] = 'empty'
    for name in examples_data.index:
        examples_data['best_cluster_nos_title'].loc[name] = st_v_clus[name]
    
    
    compare_examples=examples_data[['NOS Title','best_cluster_nos_title',
                                    'best_cluster_nos']]

    best_clusters_dict = {}
    for ix in compare_examples.index:
        keywords = new_top_terms_dict[ix]
        best_clusters_dict[ix] = []
        for k in keywords:
            best_clusters_dict[ix].append((k, feat_dict[k]))
    compare_examples = compare_examples.join(pd.DataFrame.from_dict(
            best_clusters_dict, orient = 'index'))
    compare_examples.to_csv(''.join(['/Users/stefgarasto/Google Drive/Documents/',
                        'results/NOS/progression_pathways/nos_and_skills_clusters/',
                        'sample_nos_and_skills_clusters1_',
                        '{}_{}_{}_{}_{}_{}_{}.csv'.format(
                                'titles',
                                paramsn['ngrams'],
                                'th2',
                                paramsn['tfidf_min'],
                                paramsn['tfidf_max'],
                                pofs_titles,
                                USE_WEIGHTED)]))
    
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
# checkpoint
GET_REQUIREMENTS = False
if GET_REQUIREMENTS:
    #%%
    # ### Assign each job advert to a skill cluster    
    # how are we going to match them?
    KEY = 'socs+clusters'
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
    
    unmatched_socs = set(df_nos_select['SOC4'].value_counts().index) - set(
            list(cols_v_occ.keys()))
    unmatched_socs = list(unmatched_socs - set(matches_oobsoc_to_soc2.keys()))
    # select NOS with no SOC or with "unmacthed" SOC
    nosoc = (df_nos_select['SOC4'].isnull()) | (df_nos_select['SOC4'].map(
            lambda x: x in unmatched_socs))
    
    # for the rows without SOC code, join by skill clusters
    if KEY == 'socs+clusters':
        df_nos_select['tmp'] = df_nos_select[keycols[0]].map(lambda x: str(x) + '+'
                ) + df_nos_select[keycols[1]]
        # if a specific combo is not in the job advert data as well, 
        # just use the SOC code
        flag = df_nos_select['tmp'].map(lambda x: x not in cols_v_occ_and_clus)
        #flag = flag & (~nosoc)
        df_nos_select['tmp'][flag & (~nosoc)] = df_nos_select['SOC4'][flag & (~nosoc)]
        for col in cols2match:
            # first match by skill cluster for those that don't have a nos.
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
            
        del df_nos_select['tmp']
    else:
        for col in cols2match:
            df_nos_select[col] = df_nos_select[keycol].map(
                                    lambda x: map_nos_to_req_dist(x,col,keydict))
        
    print(time.time()-t0)
    


# In[232]:

SAVEFIG = False


# In[225]:
# checkpoint
LOAD_CLUSTERS= False
'''
Here is where I take a sub selection of NOS / suites from all the possible
NOS in the super-suites.

'''
if LOAD_CLUSTERS:
    STRATEGY = 'tfidf' #'tfidf' #'we'
    pofs_clusters = 'n' #'nv' #'n'
    
    #TODO: select the appropriate file from where to get the clusters
    
    clusters2use_f = ''.join(['/Users/stefgarasto/Google Drive/Documents/results/NOS/',
                    'nlp_analysis/nos_clusters_{}_final_no_dropped/'.format(pofs_clusters),
                    'all_nos_cut_clusters_select_in_engineering_',
                    'postjoining_final_no_dropped_uni_{}_{}.xlsx'.format(
                                            STRATEGY,pofs_clusters)])
    sheet = 'hierarchicaltfidfa' #'hierarchicaltfidfa' #'hierarchical' #'hierarchicalward'
    #%
    clusters2use = pd.read_excel(clusters2use_f, sheet_name= sheet).T
    
    clusters2use = clusters2use[~clusters2use[0].isnull()]
    
    rename_dict = {}
    
    for ix in range(4):
        rename_dict[ix] = clusters2use[ix].iloc[0]
    clusters2use = clusters2use.rename(columns = rename_dict)
    clusters2use = clusters2use[1:]
    clusters2use = clusters2use.set_index(rename_dict[0])
    
    #%%
    if sheet == 'kmeans':
        labels_f = ''.join(['/Users/stefgarasto/Google Drive/Documents/results/NOS/'
                            'nlp_analysis/nos_clusters_{}_final_no_dropped/'.format(pofs_clusters),
                            'all_nos_cut_labels_kmeans_in_engineering_postjoining_'
                            'final_no_dropped_uni_{}.csv'.format(STRATEGY)])
        nos_clusters = pd.read_csv(labels_f)
        nos_clusters = nos_clusters.rename(columns = {'kmeans': 'labels'})
    elif sheet == 'hierarchical':
        labels_f = ''.join(['/Users/stefgarasto/Google Drive/Documents/results/NOS/'
                            'nlp_analysis/nos_clusters_{}_final_no_dropped/'.format(pofs_clusters),
                            'all_nos_cut_labels_in_engineering_postjoining_'
                            'final_no_dropped_uni_{}.csv'.format(STRATEGY)])
        nos_clusters = pd.read_csv(labels_f)
        nos_clusters = nos_clusters.rename(columns = {'hierarchical': 'labels'})

    elif sheet == 'hierarchicalward':
        labels_f = ''.join(['/Users/stefgarasto/Google Drive/Documents/results/NOS/'
                            'nlp_analysis/nos_clusters_{}_final_no_dropped/'.format(pofs_clusters),
                            'all_nos_cut_labels_in_engineering_postjoining_'
                            'final_no_dropped_uni_{}_w.csv'.format(STRATEGY)])
        nos_clusters = pd.read_csv(labels_f)
        nos_clusters = nos_clusters.rename(columns = {'hierarchical': 'labels'})
    elif sheet == 'hierarchicaltfidfa':
        labels_f = ''.join(['/Users/stefgarasto/Google Drive/Documents/results/NOS/'
                            'nlp_analysis/nos_clusters_{}_final_no_dropped/'.format(pofs_clusters),
                            'all_nos_cut_labels_in_engineering_postjoining_'
                            'final_no_dropped_uni_{}_a.csv'.format(STRATEGY)])
        nos_clusters = pd.read_csv(labels_f)
        nos_clusters = nos_clusters.rename(columns = {'hierarchical': 'labels'})
    elif sheet == 'hierarchicaltfidfa2':
        labels_f = ''.join(['/Users/stefgarasto/Google Drive/Documents/results/NOS/'
                            'nlp_analysis/nos_clusters_{}_final_no_dropped/'.format(pofs_clusters),
                            'all_nos_cut_labels_in_engineering_postjoining_'
                            'final_no_dropped_uni_{}_a2.csv'.format(STRATEGY)])
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
    # do PCA, jsut in case it's useful for later
    allvals = extract_vals(df_nos_select,all_edu,all_exp)
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
# checkpoint
DO_CLUS = False


if DO_CLUS:
    A = df_nos_select['Salary'].map(lambda x: take_prc(x,98))
    xlimU = max(A)
    xlimL = 0
    output_dir2 = output_dir + '/nosclusters/{}_{}_{}'.format(
                sheet,pofs_clusters, paramsn['pofs'])        
    USE_PCA = True
    if USE_PCA:
        output_dir2 += '_pca'
    for MODE in clusters2use.index:
        
        SELECT_MODE = clusters2use.loc[MODE]
        
        ''' Here, do the plots for when I take the NOS from the clusters'''

        # select the cluster
        final_nos, final_groups, larger_suites, cluster_name, cluster_name_save, \
            cluster_name_figs = select_subdf(SELECT_MODE, clusters2use, 
                                             nos_clusters,df_nos_select)
        
        # replace oob soc codes
        final_nos['SOC4'] = final_nos['SOC4'].map(replace_oob_socs)
        # load the clustering of the NOS inside
        #nos_clus_label = pd.read_csv(os.path.join(lookup_dir2, 
        #                         'all_nos_cut_labels_in_{}_{}_{}.csv'.format(
        #    cluster_name_save,qualifier,'uni')))
        #nos_clus_label = nos_clus_label.set_index('index')
        #final_nos = final_nos.join(nos_clus_label['hierarchical'])
        #print('!!!!', cluster_name_figs, final_nos['supersuite'].value_counts())
        #continue 
        #%
        # first, compute more quantities
        #final_small = final_nos[['NOS Title','myEdu','myEdu-peak','myExp','myExp-peak',
        #                 'Salary-peak','converted_skills','hierarchical']]
        # remove NOS with legacy in the title
        print('nb with legacy nos:',len(final_nos))
        final_nos = final_nos[final_nos['NOS Title'].map(lambda x: 'legacy' not in x)]
        final_nos = final_nos[final_nos.index.map(lambda x: not x[-5:]=='l.pdf')]
        print('nb without legacy nos:',len(final_nos))
        final_nos['Salary-peak'] = final_nos['Salary-peak'].map(lambda x: np.float32(x))
        # if too few NOS are left, continue
        if len(final_nos)<10:
            continue
        
        #% get number of NOS clusters
        #nb_nos_clus = len(nos_clus_label['hierarchical'].value_counts())
        # get different dataframes for each NOS cluster, only if the number of 
        # NOS is "right" (not too big, not too small)
        #min_nos = 4
        #max_nos = 50
        # extract "good" NOS clusters
        #nos_clus_dfs = []
        #for i in range(1,nb_nos_clus+1):
        #    tmp_df = final_nos[final_nos['hierarchical'] == i]
        #    if (len(tmp_df)<=max_nos) & (len(tmp_df)>=min_nos):
        #        nos_clus_dfs.append(tmp_df)
            #if i == nb_nos_clus-1:
            #    tmp_df = final_nos[final_nos['hierarchical'] == nb_nos_clus]
            #    if (len(tmp_df)<=max_nos) & (len(tmp_df)>=min_nos):
            #        nos_clus_dfs.append(tmp_df)
        #print('Number of NOS clusters selected is: ',len(nos_clus_dfs))
                
        #% extract skills that are important for each of the exp/edu pairs
        skillsdf =extract_top_skills(final_nos,all_exp,all_edu)
        skillsdf.to_csv(output_dir2 + '/NOS_topskills_for_{}_{}_v0.csv'.format(
                    cluster_name_save,KEY))

        # heatmap of exp/edu in NOS ordered by centroid
        HM = True
        if HM:
            #% plot ordered heatmaps for the whole suites cluster
            h = 15*len(final_nos)/23
            w = 12*max([len(t) for t in final_nos['NOS Title']])/70
            # education
            #print('Plotting education for all')
            #plot_centroids_hm(final_nos,w,h, cent_col = 'centEdu',qual_col = all_edu,
            #              xcat = 'Education category', 
            #              cluster_name_figs = cluster_name_figs,
            #              cluster_name_save = cluster_name_save, KEY= KEY,
            #              output_dir = output_dir2, SAVEFIG = SAVEFIG)
            # experience
            print('Plotting experience for all')
            plot_centroids_hm(final_nos,w,h, cent_col = 'centExp',qual_col = all_exp,
                          xcat = 'Experience category', 
                          cluster_name_figs = cluster_name_figs,
                          cluster_name_save = cluster_name_save, KEY= KEY,
                          output_dir = output_dir2, SAVEFIG = SAVEFIG)
            
            ## redo the heatmaps just for the NOS in each cluster
            #for ix,nos_clus_df in enumerate(nos_clus_dfs):
            #    print('Plotting education for cluster {}'.format(ix))
            #    # education
            #    h0 = 15*len(nos_clus_df)/23
            #    w0 = 12*max([len(t) for t in nos_clus_df['NOS Title']])/70
            #    plot_centroids_hm(nos_clus_df,w0,h0, cent_col = 'centEdu',qual_col = all_edu,
            #              xcat = 'Education category', 
            #              cluster_name_figs = 'Cluster {} ({})'.format(ix,cluster_name_figs),
            #              cluster_name_save = cluster_name_save + '_cl{}'.format(ix),
            #              KEY= KEY)
            #    
            #    print('Plotting experience for cluster {}'.format(ix))
            #    # experience
            #    plot_centroids_hm(nos_clus_df,w0,h0, cent_col = 'centExp',qual_col = all_exp,
            #              xcat = 'Experience category', 
            #              cluster_name_figs = 'Cluster {} ({})'.format(ix,cluster_name_figs),
            #              cluster_name_save = cluster_name_save + '_cl{}'.format(ix),
            #              KEY= KEY)
         
        SP = False
        if SP:
            #% 2D swarm plot
            plot_swarm_nos(final_nos, SALARY = True, 
                           cluster_name_save = cluster_name_save, KEY = KEY,
                           title = 'Requirements for NOS in {}'.format(
                                   cluster_name_figs), output_dir = output_dir2,
                                   SAVEFIG = SAVEFIG)
            
            #for ix, nos_clus_df in enumerate(nos_clus_dfs):
            #    plot_swarm_nos(nos_clus_df, SALARY = True,  KEY = KEY,
            #               cluster_name_save = cluster_name_save + '_cl{}'.format(ix))
    
        KM = True
        KM_MODE = 'gmm'
        KMfunc = {'km': do_kmean, 'gmm': do_gmm, 'bgmm': do_bgmm}
        nos_levels = []
        if KM:
            #for ix, nos_clus_df in enumerate(nos_clus_dfs):
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
                #if KM_MODE in ['gmm','bgmm']:
                #    X = pca.transform(clusterer.means_)[:,0]
                #elif KM_MODE == 'km':
                #    X = pca.transform(clusterer.cluster_centers_)[:,0]
                #Xsort = np.argsort(X)
                
            #% now save the relevant info
            exp2num = {'Entry-level':1, 'Mid-level': 2,'Senior-level':3}
            edu2num = {'Pregraduate':1, 'Graduate': 2,'Postgraduate':3}
            rename_dict = {'myEdu-peak': 'Educational requirement',
                           'myExp-peak':'Experience requirement',
                           'Salary-peak': 'Avg salary',
                           'NOS Title': 'NOS titles', 'URN': 'URN',
                           'converted_skills': 'Top skills',
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
            nos_groups = nos_groups.agg(np.max)
            # average salary
            nos_groups2 = group_levels['Salary-peak']
            nos_groups2 = nos_groups2.agg(np.median).map(np.round)
            nos_groups = nos_groups.join(nos_groups2)
            # NOS titles
            final_nos['NOS Title'] = final_nos['NOS Title'].map(
                    lambda x: x.capitalize())
            nos_groups2 = group_levels['NOS Title']
            nos_groups2 = nos_groups2.apply('\n '.join)
            nos_groups = nos_groups.join(nos_groups2)
            # URNs
            nos_groups2 = group_levels['URN']
            nos_groups2 = nos_groups2.apply('\n '.join)
            nos_groups = nos_groups.join(nos_groups2)
            # top 10 skills
            nos_groups2 = group_levels['converted_skills'].agg(
                    'sum').map(lambda x:x.most_common()).map(lambda x: x[:10]).map(
                            lambda x: [t[0].capitalize() for t in x]).map('\n '.join)
            nos_groups = nos_groups.join(nos_groups2)
            # top 10 job titles
            nos_groups2 = group_levels['title_processed'].agg(
                    'sum').map(lambda x:x.most_common()).map(lambda x: x[:10]).map(
                            lambda x: [t[0].capitalize() for t in x]).map('\n '.join)
            nos_groups = nos_groups.join(nos_groups2)
            # most common occupation
            nos_groups2 = group_levels['SOC4']
            nos_groups2 = nos_groups2.agg(Counter).map(lambda x: x.most_common()
                ).map(lambda x: x[:3]).map(lambda x: [socnames_dict[t[0]].capitalize() 
                for t in x])
            #nos_groups2.agg(np.max).map(lambda x: socnames_dict[x])
            nos_groups = nos_groups.join(nos_groups2)
            # top 10 keywords
            # concatenate tokes
            tokens_concat = group_levels['pruned_lemmas'].agg(sum)
            # take transform
            tfidfm_tmp = tfidf_n.transform(pd.DataFrame(tokens_concat)[
                    'pruned_lemmas']).todense()
            nos_groups2 = {}
            ix=0
            for name, group in group_levels:
                top_ngrams, top_weights, top_features = extract_top_features(
                        tfidfm_tmp[ix,:], feature_names_n, N=10)
                tmp = '\n '.join(['({}, {:.3f})'.format(top_features[ix], 
                                  top_weights[ix]) for ix in range(10)])
                nos_groups2[name] = tmp
                ix +=1
            nos_groups= nos_groups.join(pd.DataFrame.from_dict(nos_groups2, orient = 'index',
                                                     columns = ['Top keywords']))
            # most common skills cluster (?)
            nos_groups2 = group_levels['best_cluster_nos']
            nos_groups2 = nos_groups2.agg(Counter).map(lambda x: x.most_common()
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
            nos_groups.to_csv(output_dir2 + '/NOS_levels_for_{}_{}_v0.csv'.format(
                    cluster_name_save,KEY))
            
            
            
        #% for each cluster plot all NOS in order of increasing salary
        SAL = True
        if SAL:
            #for ix, nos_clus_df in enumerate(nos_clus_dfs):
            tmp_nos = final_nos.sort_values(by='Salary-peak')[['NOS Title','Salary']]
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
                hm2plot = pd.crosstab(final_nos['SOC4'].map(lambda x : socnames_dict[x]),
                                      final_nos['best_cluster_nos']).T
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
# checkpoint
DO_SUPERS = False
SAVEFIG = False
#%% first, plots for all supersuites
if DO_SUPERS:
    for SELECT_MODE in ['financialservices']: #['engineering','construction','management','financialservices']:
        
        #%
        '''
        Figure plotting for each supersuite.
        
        1. Heatmap of qualification vs experience requirements for the whole of the
        super-suite
        2. Salary box plots for the biggest suites in the super-suite
        3. Distribution of skills clusters per NOS across the super-suites (?)
        
        '''
        
        # get the data for this suite
        final_nos, final_groups, larger_suites, cluster_name, cluster_name_save, \
            cluster_name_figs = select_subdf(SELECT_MODE, [], 
                                             [], df_nos_select)
            
        # replace oob soc codes
        final_nos['SOC4'] = final_nos['SOC4'].map(replace_oob_socs)
        
        #% Replot distribution of SOC codes for suites in this super-suite
        fig = plt.figure(figsize = (12,8))
        tmp = final_nos['SOC4'].value_counts()
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
        
        #%
        # 1. Heatmap of qualification vs experience requirements for the whole of the
        # engineering super-suite
        #fig = plt.figure(figsize = (4,4))
            
        #%
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
         
        skillsdf = extract_top_skills(final_nos,all_exp,all_edu)
        skillsdf.to_csv(output_dir + '/supersuites/NOS_topskills_for_{}_{}_v2.csv'.format(
                    cluster_name_save,KEY))
            
        #%%
        # 2. Salary box plots for the biggest suites in engineering
        salary_by_suite = {}
        for suite in larger_suites:
            A = final_groups.get_group(suite)['Salary']
            A = A.values
            A = np.concatenate(tuple(A))
            A = A[~np.isnan(A)]
            salary_by_suite[suite.capitalize()] = A
            
        #%
        #salary_by_suite = pd.DataFrame.from_dict(salary_by_suite)
        t0 = time.time()
        df = pd.DataFrame({k:pd.Series(v) for k,v in salary_by_suite.items()})
        print_elapsed(t0, 'make dataframe for salaries')
        
        #%
        
        fig, ax = plt.subplots(figsize = (12,8))
        #sns.boxplot(x = 'Exp3-peak',y='MeanSalary-peak', data=eng_nos, 
        #            order = ['Entry-level','Mid-level','Senior-level'])
        
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
    
        # In[249]:
        if SELECT_MODE == 'engineering0':
            suite2use = 'rail engineering' #'down stream gas' #'rail engineering'
            #'aeronautical engineering suite 3' #(2) #'down stream gas'
            #multi utility network construction (4), down stream gas (3)
            #'aeronautical engineering suite 3'
            group = final_groups.get_group(suite2use)
            group = group.rename(columns = {'myEdu-peak': 'Qualification requirements'})
            fig = plt.figure(figsize=(8,8))
            sns.swarmplot(data = group, 
                            x = 'myExp-peak', y ='Salary-peak', hue = 'Qualification requirements',
                          order = ['Entry-level', 'Mid-level','Senior-level'], 
                          palette = nesta_colours[:2])
            #               x_bins = ['Entry-level', 'Mid-level','Senior-level'],
            #               y_bins = ['Pregraduate','Graduate','Postgraduate'])
            plt.ylabel('Average salary', fontsize = 18)
            plt.xlabel('Experience requirements', fontsize = 18)
            plt.title('Requirements for NOS in {} (\'{}\')'.format(
                    suite2use.capitalize(),cluster_name_figs), fontsize = 18)
            plt.tight_layout()
            #plt.ylim([35000,45000])
            if SAVEFIG:
                plt.savefig(output_dir+'/supersuites/NOS_progression_pathway_for_{}_{}_v2.png'.format(
                        cluster_name_save,KEY), bbox_inches='tight')
                plt.close(fig)
        
        #%%
        # distribution of skill clusters in the NOS belonging to each super-suite
        fig = plt.figure(figsize = (14,7))
        tmp = final_nos['best_cluster_nos'].value_counts()
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
        hm2plot = pd.crosstab(final_nos['SOC4'].map(lambda x : socnames_dict[x]),
                                  final_nos['best_cluster_nos']).T
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
            
        # In[ ]:
        # how many SOCs per suite?
        nb_of_socs = []
        for name,group in final_groups:
            print(name)
            print(group['SOC4'].value_counts())
            nb_of_socs.append(len(group['SOC4'].value_counts()))
            print('-'*30)
        
        # multi utility network construction (4), down stream gas (3)
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
        plt.close('all')

#%% 
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

The dataframe to save is df_nos_select

# possibly save as json?
#Might take less space

'''
rel_cols = ['NOS Title', 'URN', 'Original URN', 'Overview',
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
       'converted_skills', 'London']
SAVEAUG = False
if SAVEAUG:
    df_nos_select[rel_cols].to_csv(output_dir + 
                 '/augmented_info_NOS_in_supersuites.csv')




#%%
# ### Collect some examples

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


