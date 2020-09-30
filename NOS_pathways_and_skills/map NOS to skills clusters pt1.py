#!/usr/bin/env python
# coding: utf-8

# In[1]:

FIRST_RUN = False
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
    
    # In[3]:
    
    
    # set up plot style
    print(plt.style.available)
    plt.style.use(['seaborn-darkgrid','seaborn-poster','ggplot'])
    
    
    # In[11]:
    
    
    qualifier = 'postjoining_final_no_dropped'
    qualifier0 = 'postjoining_final_no_dropped'
    pofs = 'nv'
    
    
    # In[12]:
    
    output_dir = '/Users/stefgarasto/Google Drive/Documents/results/NOS/progression_pathways/'
    #output_dir = '/Users/stefgarasto/Google Drive/Documents/results/NOS/nlp_analysis/'
    
    
    # In[13]:
    
    
    lookup_dir= '/Users/stefgarasto/Google Drive/Documents/results/NOS/extracted/'
    lookup_dir2= '/Users/stefgarasto/Google Drive/Documents/results/NOS/nlp_analysis/'
    
    #Loading a pre-trained glove model into gensim
    # model should have already been loaded in bg_load_prepare_and_run. 
    # If not, load it here
    WHICH_GLOVE = 'glove.6B.100d' #'glove.6B.100d' #'glove.840B.300d', 
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
    
    
    # In[16]:
    
    
    #Get the NOS data 
    df_nos = pd.read_pickle(lookup_dir + 'all_nos_input_for_nlp_{}.zip'.format(qualifier0))
    
    # load the cleaned and tokenised dataset
    df_nos = df_nos.join(pd.read_pickle(lookup_dir + 
                        'all_nos_input_for_nlp_{}_pruned_{}.zip'.format(qualifier,pofs)))
    print('Done')
    
    
    # In[17]:
    
    
    # manually remove "k"s and "p"s from the pruned columns
    def remove_pk(x):
        return [t for t in x if t not in ['k','p']]
    df_nos['pruned'] = df_nos['pruned'].map(remove_pk)
    
    
    #
    
    #%%
    def replace_oob_socs(x):
        if not np.isnan(x):
            if x in matches_oobsoc_to_soc2:
                x = matches_oobsoc_to_soc2[x]
        return x
    
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
        # to matche one of the dictionary keys
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
    ''' 
    '''   
    #%%
            
    def assign_supersuite(x):
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
        
        # if the average is needed, compute it and overwrite the matrix. Note that the 
        # step above is still needed to
        # initialise the tfidf transform with the proper features and stopwords
        if (params['bywhich'] == 'suites') and (params['mode'] =='meantfidf'):
            row_names = df_nos_select['One_suite'].value_counts().index.values
            tfidfm = scipy.sparse.lil_matrix(np.zeros((len(row_names),
                                            len(feature_names)), dtype = np.float32))
            for name, group in df_nos_select.groupby('One_suite'):
                tmp = get_mean_tfidf(group[col], tfidf)
                tfidfm[igroup] = tmp
    
        feature_names = tfidf.get_feature_names()
        print_elapsed(t0, 'computing the feature vector')
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
    
    
    # In[19]:
    
    
    # Load stopwords
    with open(lookup_dir + 'stopwords_for_nos_{}_{}.pickle'.format(qualifier,pofs),'rb') as f:
        stopwords0, no_idea_why_here_stopwords, more_stopwords = pickle.load(f)
    stopwords = stopwords0 + no_idea_why_here_stopwords 
    stopwords += tuple(['¤', '¨', 'μ', 'บ', 'ย', 'ᶟ', '‰', '©', 'ƒ', '°', '„'])
    stopwords0 += tuple(['¤', '¨', 'μ', 'บ', 'ย', 'ᶟ', '‰', '©', 'ƒ', '°', '„',
                         "'m", "'re", '£'])
    stopwords0 += tuple(set(list(df_nos['Developed By'])))
    stopwords0 += tuple(['cosvr'])
    
    
    # In[20]:
    
    
    # create another column where the texts are lemmatised properly
    t0 = time.time()
    df_nos['pruned_lemmas'] = df_nos['tagged_tokens'].map(lambda x: lemmatise_pruned(x,pofs))
    print(time.time()-t0)
    
    
    # ### Only keep NOS from a super-suite
    
    # In[21]:
    
    
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
    
    
    
    #
    
    
    params = {}
    params['ngrams'] = 'uni'
    params['pofs'] = 'nv'
    params['tfidf_min'] = 3
    params['tfidf_max'] = 0.5
    
    params['bywhich'] = 'docs' #'docs' #'suites'
    params['mode'] = 'tfidf' #'tfidf' #'meantfidf' #'combinedtfidf' #'meantfidf'
    
    
    # In[43]:
    
    
    # define the transform: this one can easily be the same for both 
    # keywords and the clustering
    tfidf = define_tfidf(params, stopwords0)
    
    
    # ### Check keywords at the NOS level
    # We can take a look at some of the terms with highest tf-idf score in each NOS
    
    # In[44]:
    
    
    SAVEKW= False
    
    
    # In[45]:
    
    
    # get the features
    tfidfm, feature_names, tfidf, textfortokens = get_tfidf_matrix(
            params, df_nos_select, tfidf, col = 'pruned_lemmas')
    
    
    # In[47]:
    
    
    print('Number of features: {}'.format(len(feature_names)))
    N = 2000
    print('Some features:')
    print(feature_names[N:N+100])
    
    
    
    # In[77]:
    
    
    top_terms_dict = {}
    top_keywords_dict = {}
    #for name, group in ifa_df.groupby('Route'):
    igroup = 0
    n_keywords =[]
    n_repeated = []
    #top_terms = {}
    t0 = time.time()
    tfidfm_dense = tfidfm.todense()
    for ix,name in enumerate(df_nos_select.index):
        #top_terms = get_top_words(df_nos_select.loc[name]['pruned'], feature_names, tfidf, n = 20)
        top_ngrams = np.argsort(tfidfm_dense[ix,:])
        top_ngrams = top_ngrams.tolist()[0][-20:]
        top_ngrams = top_ngrams[::-1]
        # only retain the ones with non zero features
        top_ngrams = [elem for elem in top_ngrams if tfidfm_dense[ix,elem]>0]    
        top_features = [feature_names[elem] for elem in top_ngrams]
        top_terms_dict[name] = {}
        top_terms_dict[name] = top_features
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
    tfidfm_dense = None
    
    
    # In[78]:
    
    
    # just to check results
    '''
    print(list(top_terms_dict.keys())[885:887])
    top_terms_weights = get_top_words_weights([df_nos_select.iloc[0]['pruned_lemmas']], feature_names, tfidf, n = 20)
    print(top_terms_weights.sort_values(by = 'tfidf', ascending = False).head(n=20))
    '''
    # note that the get_top_words_weights function is probably wrong - but it doesn't matter now
    print('not now')
    
    
    # In[82]:
    
    
    # remove top terms that are not in the chosen gensim model
    new_top_terms_dict = {}
    for k,v in top_terms_dict.items():
        # check if the top terms for each document are in the gensim model
        new_top_terms = prep_for_gensim(v, model)
        # only retains the ones in the model
        new_top_terms_dict[k] = new_top_terms
        if np.random.randn(1)>3:
            print(k, new_top_terms, len(new_top_terms), len(v))
    
    #%% 
    ''' Create skills clusters '''
    # create skill clusters
    clus_names, comparison_vecs, skill_cluster_vecs = load_and_process_clusters(model)
    
    
    # In[192]:
    # ### Assign each NOS to a skill cluster
    
    '''
    Link each NOS to a skill cluster
    '''
    
    st_v_clus = {}
    counter = 0
    for ix,k in enumerate(new_top_terms_dict):
        test_skills,_ = get_mean_vec(new_top_terms_dict[k], model)
        st_v_clus[k] = highest_similarity(test_skills, comparison_vecs, clus_names)
        
    
    # In[ ]:
    
    
    # add the best clusters to the nos dataframe
    tmp = pd.DataFrame.from_dict(st_v_clus, orient = 'index')
    tmp = tmp.rename(columns = {0: 'best_cluster_nos'})
    df_nos_select['best_cluster_nos'] = tmp['best_cluster_nos']
    
    
    #%%
    # replace out-of-boundary SOCs
    
    
    #%% load the dictionaries mapping SOC and SC to requirements if necessary
    if not 'cols_v_clus' in locals():
        with open(os.path.join(saveoutput,'cols_v_clus.pickle'), 'rb') as f:
            cols_v_clus = pickle.load(f)
    
    #% SOC
    if not 'cols_v_occ' in locals():
        with open(os.path.join(saveoutput,'cols_v_occ.pickle'), 'rb') as f:
            cols_v_occ = pickle.load(f)
        
    #% SOC+cluster
    if not 'cols_v_occ_and_clus' in locals():
        with open(os.path.join(saveoutput,'cols_v_occ_and_clus.pickle'), 'rb') as f:
            cols_v_occ_and_clus = pickle.load(f)
    
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
    
    
    
    
    #%%
    
    def select_subdf(SELECT_MODE, clusters2use,suites_clusters,df_nos_select):
        if isinstance(SELECT_MODE, str):
            tmp_dict = {'engineering': 'Engineering', 'management': 'Management',
                        'financialservices': 'Financial services', 
                        'construction': 'Construction'}
            # select NOS from super suite
            cluster_name = SELECT_MODE
            cluster_name_save = cluster_name
            cluster_name_figs = tmp_dict[SELECT_MODE]
            subset_nos = df_nos_select[df_nos_select['supersuite']== SELECT_MODE]
        elif isinstance(SELECT_MODE, int):
            cluster_name = clusters2use[SELECT_MODE][1]
            cluster_name_save = cluster_name.replace(' ','_')
            cluster_name_figs = cluster_name.capitalize()
            suites2use = list(suites_clusters[suites_clusters['hierarchical'].map(
                    lambda x: x in clusters2use[SELECT_MODE][0])]['Suite_names'].values)
            subset_nos = df_nos_select[df_nos_select['One_suite'].map(
                    lambda x: x in suites2use)]
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
        all_lengths[::-1].sort()
        print(all_lengths)
        #th_supers = ['engineering': 40, 'financialservices': ]
        for name, group in final_groups:
            if isinstance(SELECT_MODE, int):
                larger_suites.append(name)
            elif len(group)> all_lengths[15]:#th_supers[SELECT_MODE]:
                #print(name, len(group))
                larger_suites.append(name)
    
        return final_nos, final_groups, larger_suites, cluster_name,  \
                        cluster_name_save, cluster_name_figs
    
    #%%
    def extract_top_skills(final_nos,all_exp,all_edu):
        skillsdf = []
        N = 20
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
    
    #%% Set some things up
    all_edu = ['Pregraduate','Graduate','Postgraduate']
    all_exp =  ['Entry-level','Mid-level','Senior-level']
    
    edu_colours = {'Pregraduate': nesta_colours[0], 'Graduate': nesta_colours[1],
                           'Postgraduate': nesta_colours[3]}
    exp_colours = {'Entry-level': nesta_colours[4], 'Mid-level': nesta_colours[6],
                           'Senior-level': nesta_colours[8]}
    
    #%%
    def plot_centroids_hm(final_small,w,h, cent_col = 'centEdu',qual_col = all_edu,
                          xcat = 'Education category', cluster_name_figs = '',
                          cluster_name_save = '', KEY='', output_dir = output_dir):
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
        if cluster_name_save in ['engineering','construction','management','financialservices']:
            output_dir += '/supersuites'
        else:
            output_dir += '/nosclusters'
        if SAVEFIG:
            plt.savefig(output_dir + '/NOS_cent_{}_ordered_for_{}_{}.png'.format(
                    xcat_red[:3],cluster_name_save,KEY), bbox_inches='tight')
            plt.close(fig)
            
    #%%
    def plot_swarm_nos(final_nos, SALARY = True, cluster_name_save = '', KEY = '',
                       title = '', output_dir = output_dir):
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
        if cluster_name_save in ['engineering','construction','management','financialservices']:
            output_dir = output_dir + '/supersuites'
        else:
            output_dir += '/nosclusters'
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

    def do_gmm(xx, ks = np.arange(2,4),N=100):
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
            print_elapsed(t0,'kmeans for k={}'.format(k))
        # what number of clusters has highest stability?
        kmaxgmm1 = ks[np.array(stabgmm).argmax()]
        kmaxgmm2 = ks[np.array(bicgmm).argmin()]
        kmaxgmm3 = ks[np.array(aicgmm).argmin()]
        kmaxgmm = min([kmaxgmm1, kmaxgmm2, kmaxgmm3])
        # redo one last clustering with kmax 
        # and lots of iteration to get the stable versions
        gmm = mixture.GaussianMixture(kmaxgmm, n_init=100, 
                                      random_state = np.random.randint(1e7))
        labelsgmm = gmm.fit_predict(xx)
        #bicgmm1 = gmm.bic(xx)
        #aicgmm1 = gmm.aic(xx)
        return labelsgmm, gmm, kmaxgmm, (stabgmm, bicgmm, aicgmm)

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

# In[232]:

SAVEFIG = True

DO_SUPERS = True

#%% first, plots for all supersuites
if DO_SUPERS:
    for SELECT_MODE in ['engineering','construction','management','financialservices']:
        
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
            plt.savefig(output_dir + '/supersuites/NOS_occupations_for_{}_{}_v0.png'.format(
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
                                           cluster_name_figs))
        if SAVEFIG:
            plt.savefig(output_dir + '/supersuites/NOS_matchbysoc_exp_vs_edu_for_{}_{}_v0.png'.format(
                    cluster_name_save,KEY), bbox_inches='tight')
            plt.close(fig)
         
        skillsdf = extract_top_skills(final_nos,all_exp,all_edu)
        skillsdf.to_csv(output_dir + '/supersuites/NOS_topskills_for_{}_{}_v0.csv'.format(
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
                                    for t in [0,1,3,4,5,6,8,9]])
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
                '/supersuites/NOS_matchbysoc_full_salary_by_suites_for_{}_{}_v0.png'.format(
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
                plt.savefig(output_dir+'/supersuites/NOS_progression_pathway_for_{}_{}_v0.png'.format(
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
            plt.savefig(output_dir + '/supersuites/NOS_skills_clusters_for_{}_{}_v0.png'.format(
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
            plt.savefig(output_dir + '/supersuites/NOS_SC_vs_SOC_for_{}_{}_v0.png'.format(
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
            plt.savefig(output_dir + '/supersuites/NOS_socs_per_suites_for_{}_{}_v0.png'.format(
                    cluster_name_save,KEY), bbox_inches='tight')
            plt.close(fig)
        plt.close('all')

# In[225]:
'''
Here is where I take a sub selection of NOS / suites from all the possible
NOS in the super-suites.

Some interesting clusters are the following:
19 = network engineering (utilities)
1 = pension schemes
4+5 = investments advice
46+47 = wood
51+52 = rail engineering
53 = engineering management and maintenance
55+56 = mechanical engineering

SELECT_MODE can be either one of the super suites or the index of the suites
cluster of interest.

The old clusters are:

clusters2use = [([1], 'pension schemes'),
                ([4,5], 'investments advice'),
                ([19], 'network engineering (utilities)'), 
                ([46,47],'wood in construction'),
                ([51,52],'rail engineering'),
                ([53], 'engineering - management and maintenance'),
                ([55,56], 'mechanical engineering')]
'''
clusters2use = [([17], 'port operations'),
                ([4], 'pension schemes'),
                ([54],'rail engineering'),
                ([51], 'temperature control engineering'),
                ([18], 'food and hospitality'),
                ([21], 'entertainment industry'),
                ([31], 'social care'),
                ([46], 'network engineering (utilities)')]
#rows: 6,7,14,16,18,26,29,33

#cluster_id = 0

# first laod the suites clusters 
suites_clusters_f = ''.join(['/Users/stefgarasto/Google Drive/Documents/results/NOS/',
   'nlp_analysis/all_supersuites_hierarchical_results_postjoining_final_',
   'no_dropped_suites_combinedtfidf_uni.csv'])

# uncomment when needed to load for the first time           
suites_clusters = pd.read_csv(suites_clusters_f)
  
#%%
if FIRST_RUN:    
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
    allvals = np.array(df_nos_select[all_edu + all_exp + ['Salary-peak']].values, dtype =np.float32)
    allvals[np.isnan(allvals[:,6]),6] = np.nanmean(allvals[:,6])
    x = StandardScaler().fit_transform(allvals)
    #x = whiten(allvals)
    pca = PCA('mle')
    prinComp = pca.fit_transform(x)

    
#%%
DO_CLUS = False
A = df_nos_select['Salary'].map(lambda x: take_prc(x,98))
xlimU = max(A)
xlimL = 0

if DO_CLUS:
    for SELECT_MODE in range(0,len(clusters2use)):
        
        ''' Here, do the plots for when I take the NOS from the clusters'''

        # select the cluster
        final_nos, final_groups, larger_suites, cluster_name, cluster_name_save, \
            cluster_name_figs = select_subdf(SELECT_MODE, clusters2use, 
                                             suites_clusters,df_nos_select)
        
        # replace oob soc codes
        final_nos['SOC4'] = final_nos['SOC4'].map(replace_oob_socs)
        # load the clustering of the NOS inside
        nos_clus_label = pd.read_csv(os.path.join(lookup_dir2, 
                                 'all_nos_cut_labels_in_{}_{}_{}.csv'.format(
            cluster_name_save,qualifier,'uni')))
        nos_clus_label = nos_clus_label.set_index('index')
        final_nos = final_nos.join(nos_clus_label['hierarchical'])
        print('!!!!', cluster_name_figs, final_nos['supersuite'].value_counts())
        continue 
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
        
        
        #% get number of NOS clusters
        nb_nos_clus = len(nos_clus_label['hierarchical'].value_counts())
        # get different dataframes for each NOS cluster, only if the number of 
        # NOS is "right" (not too big, not too small)
        min_nos = 4
        max_nos = 50
        # extract "good" NOS clusters
        nos_clus_dfs = []
        for i in range(1,nb_nos_clus+1):
            tmp_df = final_nos[final_nos['hierarchical'] == i]
            if (len(tmp_df)<=max_nos) & (len(tmp_df)>=min_nos):
                nos_clus_dfs.append(tmp_df)
            #if i == nb_nos_clus-1:
            #    tmp_df = final_nos[final_nos['hierarchical'] == nb_nos_clus]
            #    if (len(tmp_df)<=max_nos) & (len(tmp_df)>=min_nos):
            #        nos_clus_dfs.append(tmp_df)
        print('Number of NOS clusters selected is: ',len(nos_clus_dfs))
                
        #% extract skills that are important for each of the exp/edu pairs
        skillsdf =extract_top_skills(final_nos,all_exp,all_edu)
        skillsdf.to_csv(output_dir + '/nosclusters/NOS_topskills_for_{}_{}_v0.csv'.format(
                    cluster_name_save,KEY))

        # heatmap of exp/edu in NOS ordered by centroid
        HM = True
        if HM:
            #% plot ordered heatmaps for the whole suites cluster
            h = 15*len(final_nos)/23
            w = 12*max([len(t) for t in final_nos['NOS Title']])/70
            # education
            print('Plotting education for all')
            plot_centroids_hm(final_nos,w,h, cent_col = 'centEdu',qual_col = all_edu,
                          xcat = 'Education category', 
                          cluster_name_figs = cluster_name_figs,
                          cluster_name_save = cluster_name_save, KEY= KEY)
            # experience
            print('Plotting experience for all')
            plot_centroids_hm(final_nos,w,h, cent_col = 'centExp',qual_col = all_exp,
                          xcat = 'Experience category', 
                          cluster_name_figs = cluster_name_figs,
                          cluster_name_save = cluster_name_save, KEY= KEY)
            
            # redo the heatmaps just for the NOS in each cluster
            for ix,nos_clus_df in enumerate(nos_clus_dfs):
                print('Plotting education for cluster {}'.format(ix))
                # education
                h0 = 15*len(nos_clus_df)/23
                w0 = 12*max([len(t) for t in nos_clus_df['NOS Title']])/70
                plot_centroids_hm(nos_clus_df,w0,h0, cent_col = 'centEdu',qual_col = all_edu,
                          xcat = 'Education category', 
                          cluster_name_figs = 'Cluster {} ({})'.format(ix,cluster_name_figs),
                          cluster_name_save = cluster_name_save + '_cl{}'.format(ix),
                          KEY= KEY)
                
                print('Plotting experience for cluster {}'.format(ix))
                # experience
                plot_centroids_hm(nos_clus_df,w0,h0, cent_col = 'centExp',qual_col = all_exp,
                          xcat = 'Experience category', 
                          cluster_name_figs = 'Cluster {} ({})'.format(ix,cluster_name_figs),
                          cluster_name_save = cluster_name_save + '_cl{}'.format(ix),
                          KEY= KEY)
         
        SP = True
        if SP:
            #% 2D swarm plot
            plot_swarm_nos(final_nos, SALARY = True, 
                           cluster_name_save = cluster_name_save, KEY = KEY,
                           title = 'Requirements for NOS in {}'.format(
                                   cluster_name_figs))
            
            #for ix, nos_clus_df in enumerate(nos_clus_dfs):
            #    plot_swarm_nos(nos_clus_df, SALARY = True,  KEY = KEY,
            #               cluster_name_save = cluster_name_save + '_cl{}'.format(ix))
    
        KM = True
        KM_MODE = 'gmm'
        KMfunc = {'km': do_kmean, 'gmm': do_gmm, 'bgmm': do_bgmm}
        nos_levels = []
        if KM:
            for ix, nos_clus_df in enumerate(nos_clus_dfs):
                #%
                vals = np.array(nos_clus_df[all_edu[:2] + all_exp[:2]],
                                                dtype =np.float32)
                #icol = vals.shape[1]
                for prc in [25,50,75]:
                    zz = nos_clus_df['Salary'].map(lambda x: take_prc(x,prc
                                    )).values
                    zz[np.isnan(zz)] = np.nanmean(zz)
                    vals = np.concatenate((vals,zz[:,np.newaxis]),axis = 1)
                    #vals[np.isnan(vals[:,6]),6] = np.nanmean(vals[:,6])
                xsmall = StandardScaler().fit_transform(vals)
                labels, clusterer, kmax, stab = KMfunc[KM_MODE](xsmall, 
                                ks = np.arange(2,min([10,len(nos_clus_df)])),N=100)
                #%
                nos_levels.append((labels,clusterer,kmax,stab))
                #if KM_MODE in ['gmm','bgmm']:
                #    X = pca.transform(clusterer.means_)[:,0]
                #elif KM_MODE == 'km':
                #    X = pca.transform(clusterer.cluster_centers_)[:,0]
                #Xsort = np.argsort(X)
                
            #%% now save the relevant info
            exp2num = {'Entry-level':1, 'Mid-level': 2,'Senior-level':3}
            edu2num = {'Pregraduate':1, 'Graduate': 2,'Postgraduate':3}
            rename_dict = {'myEdu-peak': 'Educational requirement',
                           'myExp-peak':'Experience requirement',
                           'Salary-peak': 'Avg salary',
                           'NOS Title': 'NOS titles', 'converted_skills': 'Top skills'}
            for ix, nos_clus_df in enumerate(nos_clus_dfs):
                nos_clus_df['labels'] = nos_levels[ix][0]
                nos_clus_df['NOS Title'] = nos_clus_df['NOS Title'].map(
                        lambda x: x.capitalize())
                nos_groups = nos_clus_df.groupby('labels')[['myExp-peak','myEdu-peak']]
                nos_groups = nos_groups.agg(np.max)
                nos_groups2 = nos_clus_df.groupby('labels')['Salary-peak']
                nos_groups2 = nos_groups2.agg(np.mean).map(np.round)
                nos_groups = nos_groups.join(nos_groups2)
                nos_groups2 = nos_clus_df.groupby('labels')['NOS Title']
                nos_groups2 = nos_groups2.apply('; '.join)
                nos_groups = nos_groups.join(nos_groups2)
                nos_groups2 = nos_clus_df.groupby('labels')['converted_skills'].agg(
                        'sum').map(lambda x:x.most_common()).map(lambda x: x[:10]).map(
                                lambda x: [t[0].capitalize() for t in x]).map('; '.join)
                nos_groups = nos_groups.join(nos_groups2)
                # now add the skills 
                nos_groups['myExp-num'] = nos_groups['myExp-peak'].map(lambda x: exp2num[x])
                nos_groups['myEdu-num'] = nos_groups['myEdu-peak'].map(lambda x: edu2num[x])
                nos_groups = nos_groups.sort_values(by=['myEdu-num','myExp-num','Salary-peak'])
                nos_groups = nos_groups[list(rename_dict.keys())]
                nos_groups = nos_groups.rename(columns = rename_dict)
                # save these potential pathways
                nos_groups.to_csv(output_dir + '/nosclusters/NOS_levels_for_{}_{}_v0.csv'.format(
                        cluster_name_save+'_cl{}'.format(ix),KEY))
            
            
            
        #%% for each cluster plot all NOS in order of increasing salary
        SAL = True
        if SAL:
            for ix, nos_clus_df in enumerate(nos_clus_dfs):
                tmp_nos = nos_clus_df.sort_values(by='Salary-peak')[['NOS Title','Salary']]
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
                    plt.savefig(output_dir + 
                        '/nosclusters/NOS_ordered_by_salary_for_{}_{}_v0.png'.format(
                        cluster_name_save + '_cl{}'.format(ix),KEY), bbox_inches='tight')
                    plt.close(fig)
        
        #%%
        # plot heatmap of skills clusters vs occupations
        SCVSSOC = True
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
                plt.savefig(output_dir +
                        '/nosclusters/NOS_skills_clusters_for_{}_{}_v0.png'.format(
                        cluster_name_save,KEY))
                plt.close(fig)
            
        #%%


#TODO: improve 2Dscatter plot (at the very list change color map and check salary=0)


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


