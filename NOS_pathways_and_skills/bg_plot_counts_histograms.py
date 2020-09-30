#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 16:45:11 2019

@author: stefgarasto


NOTES:
This script only plots the histogram counts for certain columns in the BG data
For now it is setup to load data from the job ads dataset across all years
between 2012 and 2018. The counts were computed separately and saved.
"""
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import seaborn as sns
'''

#%%
#from utils_bg import nesta_colours, nesta_colours_combos

#%%
#plt.style.use(['seaborn-darkgrid','seaborn-poster','ggplot'])

#%%
#from utils_bg import saveoutput, savefigures, socnames_dict

#%%
#socnames_file = '/Users/stefgarasto/Google Drive/Documents/data/ONS/' \
#    + 'soc2010indexversion705june2018.xls'
#socnames = pd.read_excel(socnames_file, sheet_name = 'SOC2010 Structure')
#print(socnames.columns)

#%%
WHOLEDATA = False
if WHOLEDATA:
    with open(os.path.join(saveoutput,'info_about_counts_in_bg_data.pickle'),
              'rb') as f:
        info_about_counts, Ntot, nullvalues = pickle.load(f)
    
    #%%
    hist_xlabels = ['Minimum experience (years)', 'Maximum experience (years)',
                    'Minimum education (years)',
                    'Maximum education (years)','Occupations (3-digits SOC codes)',
                    'SOC codes certainty']
    w0 = .5
    with sns.plotting_context('talk'):
        for ix,outputcol in enumerate(['MinExp','MaxExp','MinEdu','MaxEdu',
                                       'new_soc','soc_score']):
            key = 'Counts for {}'.format(outputcol)
            tmp = info_about_counts[key]
            nb_of_elems = len(tmp)
            if outputcol == 'new_soc':
                fig = plt.figure(figsize = (15,.21*nb_of_elems))
                tmp = tmp[::-1]
            elif 'Exp' in outputcol:
                tmp2 = info_about_counts['Percentages for {}'.format(outputcol)]
                # there are many that only have count = 1
                tmp = tmp[tmp2>.2]
                nb_of_elems = len(tmp)
                fig = plt.figure(figsize = (w0*nb_of_elems,4))
            else:
                fig = plt.figure(figsize = (max(w0*nb_of_elems,4),4))
            if outputcol == 'new_soc':
                tmp.plot('barh',color = nesta_colours[3])
                plt.draw()
                plt.ylabel(hist_xlabels[ix])
                # change soc codes to occupations
                T = plt.yticks()
                for t in T[1]:
                    try:
                        t.set_text(socnames_dict[int(t.get_text())])
                    except:
                        print(t)
                plt.yticks(T[0],T[1])
            else:
                tmp.plot('bar', color= nesta_colours[3])
                plt.draw()
                plt.xlabel(hist_xlabels[ix])
                if 'Exp' in outputcol:
                    # for experience I need to change the labels to years
                    T = plt.xticks()
                    for t in T[1]:
                        t.set_text('{:.2f}'.format(float(t.get_text())/12.0))
                    plt.xticks(T[0],T[1])
            plt.tight_layout()
            plt.savefig(os.path.join(saveoutput,
                            'histogram_counts_for_{}_all_years.png'.format(
                                    outputcol)))

#%%
GOODDATA = True
if GOODDATA:
    SUPERS = ['all','engineering','management','financialservices','construction']
    KEYS = ['reduced by ' + t for t in SUPERS]
    NEW_KEYS = ['Within all super-suites','Within \'Engineering\'', 
                'Within \'Management\'','Within \'Financial Services\'',
                'Within \'Construction\'']
    rename_dict= {}
    for ix,t in enumerate(KEYS):
        rename_dict[t] = NEW_KEYS[ix]
        
    COLOURS = nesta_colours[2:3] #nesta_colours[0:1] + nesta_colours[2:3]
    #{NEW_KEYS[0]: nesta_colours[0], NEW_KEYS[1]: nesta_colours[2]}
    for ix,t in enumerate(SUPERS[1:]):
        COLOURS.append(super_suites_colours[t])
    #    COLOURS[NEW_KEYS[2+ix]] = super_suite_colours[t]
    
    with open(os.path.join(saveoutput,
            'info_about_counts_in_bg_data_small_dataset_{}.pickle'.format(
                    target_var)),'rb') as f2:
        info_about_counts,Ns,posterior = pickle.load(f2)
    hist_xlabels = {'MinEdu': 'Minimum education (years)',
                    'MinExp': 'Minimum experience (years)',
                    'SOC': 'Occupations (4-digits SOC codes)',
                    'Edu':'Minimum education (category)',
                    'Eduv2':'Minimum education (category)',
                    'Exp': 'Minimum experience (category)',
                    'Exp3': 'Minimum experience (category)'}
    w0 = .5

#%%
cols_to_count_dict= {'Edu': ['MinEdu','SOC','Edu'],
                     'Eduv2': ['MinEdu','SOC','Eduv2'],
                     'Exp': ['MinExp','SOC','Exp','Exp3'],
                     'Exp3': ['MinExp','SOC','Exp3']}
cols_to_count = cols_to_count_dict[target_var]
SAVEFIG = True
if GOODDATA:
    for ix,outputcol in enumerate(cols_to_count):
        df = pd.DataFrame.from_dict(info_about_counts[
                'Percentages for {}'.format(outputcol)])
        nb_of_elems = len(df)
        df = df.rename(columns = rename_dict)
        #del df['all good data']
        if outputcol == 'SOC':
            with sns.plotting_context('paper'):
                # only plot the ones with some signal
                df2 = df.sort_values(by= 'Within all super-suites', 
                                     ascending = False)
                #A = df2.T.max()
                #df2 = df2[A>.1]
                df2 = df2.iloc[:50] # only select the top 50
                df2 = df2.sort_values(by= 'Within all super-suites', 
                                     ascending = True)
                nb_of_elems = len(df2)
                fig = plt.figure(figsize = (8,w0*nb_of_elems/2))
                ax = plt.gca()
                # plot
                df2.plot(y = df.columns, color = COLOURS, ax = ax, kind = 'barh')
                #df.plot(y = df.columns, color = COLOURS, ax = ax, kind = 'barh')
                # change soc codes to occupations
                T = plt.yticks()
                for t in T[1]:
                    try:
                        t.set_text(socnames_dict[int(t.get_text()[:4])])
                    except:
                        print(t)
                plt.yticks(T[0],T[1])
                plt.ylabel(hist_xlabels[outputcol])
                plt.xlabel('Percentage of adverts')
        elif outputcol in ['Edu','Eduv2']:
            df = df.reindex(['Pregraduate','Graduate','Postgraduate'])
            with sns.plotting_context('talk'):
                fig = plt.figure(figsize = (11,5))
                ax = plt.gca()
                df.plot(y = df.columns, color = COLOURS, ax = ax, kind= 'bar')
                plt.xlabel(hist_xlabels[outputcol])
                plt.ylabel('Percentage of adverts')
        else:
            if outputcol == 'MinExp':
                #df = df/12
                df = df.set_index(np.around(df.index/12,3))
                df = df[df.T.max()>.2]
                nb_of_elems = len(df)
            with sns.plotting_context('talk'):
                fig = plt.figure(figsize = (2*w0*nb_of_elems,5))
                ax = plt.gca()
                df.plot(y = df.columns, color = COLOURS, ax = ax, kind= 'bar')
                plt.xlabel(hist_xlabels[outputcol])
                plt.ylabel('Percentage of adverts')
        plt.tight_layout()           
        plt.draw()
        if SAVEFIG:
            plt.savefig(os.path.join(savefigures,
            'histogram_percentages_for_{}_all_years_relevant_data_{}2.png'.format(
                       outputcol,target_var)))
    
    # now the heatmap SOC vs category
    with sns.plotting_context('talk'):
        fig = plt.figure(figsize = (22,33))
        sns.heatmap(posterior.T[['Pregraduate','Graduate','Postgraduate']],
                    cmap = sns.cm.rocket_r)
        #sns.heatmap(posterior.T, cmap = sns.cm.rocket_r, mask = posterior.T<match_th)
        plt.xlabel('Education category', fontsize = 24)
        plt.ylabel('Occupations (4-digit level)', fontsize = 24)
        # change soc codes to occupations
        T = plt.yticks()
        if isinstance(T[1][0].get_text(),float):
            for t in T[1]:
                try:
                    t.set_text(socnames_dict[int(t.get_text()[:-2])])
                except:
                    print(t)
            plt.yticks(T[0],T[1])
        plt.tight_layout()
        
    if SAVEFIG:
        plt.savefig(os.path.join(savefigures, 
            'heatmap_soc_by_{}_category_normalised_by_soc_all2.png'.format(
                    target_var)))
        
#%%
if GOODDATA:
    for ix,outputcol in enumerate(cols_to_count):
        df = pd.DataFrame.from_dict(info_about_counts[
                'Counts for {}'.format(outputcol)])
        nb_of_elems = len(df)
        df = df.rename(columns = rename_dict)
        #del df['all good data']
        if outputcol == 'SOC':
            with sns.plotting_context('paper'):
                # only plot the ones with some signal
                df2 = df.sort_values(by= 'Within all super-suites', 
                                     ascending = False)
                #A = df2.T.max()
                #df2 = df2[A>3000]
                df2 = df2.iloc[:50]
                df2 = df2.sort_values(by= 'Within all super-suites', 
                                     ascending = True)
                nb_of_elems = len(df2)
                fig = plt.figure(figsize = (8,w0*nb_of_elems/2))
                ax = plt.gca()
                # plot
                df2.plot(y = df.columns, color = COLOURS, ax = ax, kind = 'barh')
                #df.plot(y = df.columns, color = COLOURS, ax = ax, kind = 'barh')
                # change soc codes to occupations
                T = plt.yticks()
                for t in T[1]:
                    try:
                        t.set_text(socnames_dict[int(t.get_text()[:4])])
                    except:
                        print(t)
                plt.yticks(T[0],T[1])
                plt.ylabel(hist_xlabels[outputcol])
                plt.xlabel('Advert counts')
        elif outputcol in ['Edu','Eduv2']:
            df = df.reindex(['Pregraduate','Graduate','Postgraduate'])
            with sns.plotting_context('talk'):
                fig = plt.figure(figsize = (11,5))
                ax = plt.gca()
                df.plot(y = df.columns, color = COLOURS, ax = ax, kind= 'bar')
                plt.xlabel(hist_xlabels[outputcol])
                plt.ylabel('Advert counts')
        else:
            if outputcol == 'MinExp':
                # divide indices by 12 to get years rather than months
                df = df.set_index(np.around(df.index/12,3))
                df = df[df.T.max()>1000]
                nb_of_elems = len(df)
            with sns.plotting_context('talk'):
                fig = plt.figure(figsize = (2*w0*nb_of_elems,5))
                ax = plt.gca()
                df.plot(y = df.columns, color = COLOURS, ax = ax, kind= 'bar')
                plt.xlabel(hist_xlabels[outputcol])
                plt.ylabel('Advert counts')
        plt.tight_layout()            
        plt.savefig(os.path.join(savefigures,
                 'histogram_counts_for_{}_all_years_relevant_data_{}.png'.format(
                       outputcol,target_var)))

    