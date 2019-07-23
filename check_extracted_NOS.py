#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:43:35 2019

@author: stef

This script is to analyse and produce some summary statistics for (a subset of)
all the NOS dictionaries extracted using the function scrape_nos.py.
Which dictionaries to analyse can be set using the ``which_files'' list.

"""

import os
import pickle
from collections import defaultdict
#import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


####### Definitions
def unique(list1):
    '''
    # this functions takes a list of reference numbers for NOS and returns a list
    '''
    # without repetitions and a list with the repeated ref. numbers.
    # intilize a null list
    unique_list = []
    repeated_elems = []
    repeated_indices = [] 
    # traverse for all elements
    for ix, x in enumerate(list1):
        # check if exists in unique_list or not
        if x not in unique_list:
            # this list contains all unique refs, including the ones that are repeated
            unique_list.append(x)
        else:
            if x not in repeated_elems:
                repeated_elems.append(x)
                # get where the repeated indices are to then get the file names
                tmp = [i for i,y in enumerate(list1) if y==x]
                repeated_indices.append(tmp)
    repeated_counts = np.zeros(len(repeated_elems))
    for ii,x in enumerate(repeated_elems):
        repeated_counts[ii]= len([jj for jj in list1 if jj == x])
    return unique_list, repeated_elems, repeated_counts, repeated_indices

######## Main code

data_dir = '../../results/NOS/extracted'
summary_dir = '../../results/NOS/summary'
#set which NOS dictionaries to analyse
which_files = ['extracted_standards_New NOS 1.pickle', 'extracted_standards_New NOS 2.pickle']
for ii in range(1,23):
    which_files.append('extracted_standards_Old NOS {}.pickle'.format(ii))
#which_file = 'extracted_standards_Old NOS 22.pickle'

# Write the whole summary to file if WRITEFILE is True
version_load = 'new_' #'v2_'
WRITEFILE = True
if WRITEFILE:
    fsave  =open(os.path.join(summary_dir, version_load + 'summary1.txt'), 'w')

dict_len = []
sections_list = []
sections_count = {}

total_doc_extracted = 0
total_doc_failed = 0
total_doc_unique = 0

# collect all standards that failed, the repeated ones and those with missing urn
all_standard_failed = []
all_repeated_ref = []
all_repeated_counts = np.zeros(0)
all_repeated_files = defaultdict()
all_urnmissing_files = []
all_urnmissing_names = []

texts = []
for which_file in which_files:
    texts.append('Working on: ' + which_file + '\n')

    # load the file
    # choose the prefix of the version to load
    #version_load = 'new_' #'v2_'
    with open(os.path.join(data_dir, version_load + which_file),'rb') as f:
        standard_info, standard_ref, standard_failed, standard_files = pickle.load(f)

    # OBS1: Some standard refs are the same!
    # Get the unique and repeated ones
    unique_ref, repeated_ref, repeated_counts, repeated_indices = unique(standard_ref)
    all_repeated_ref += repeated_ref
    all_repeated_counts = np.concatenate((all_repeated_counts, repeated_counts))
    for irep, rep in enumerate(repeated_ref):
        all_repeated_files[rep] = {}
        for iind1,iind2 in enumerate(repeated_indices[irep]):
            all_repeated_files[rep]['file{}'.format(iind1+1)] = which_file[
                    20:30].replace('.','') + '/' + standard_files[iind2]
    total_doc_unique += len(unique_ref)

    # for all the repeated urns also collect the NOS name and the file name (?)
    
    # total number of files in this folder:
    Ntot = len(standard_failed)+len(standard_ref)
    Nfailed = len(standard_failed)

    all_standard_failed+= standard_failed
    total_doc_extracted += len(standard_ref)
    total_doc_failed += len(standard_failed)
    # print number of actually failed extractions
    texts.append('Correct extractions: {:4f}% \n'.format(len(standard_info)/Ntot*100))
    texts.append('That is, {} files over {} failed the extraction\n'.format(Nfailed, Ntot))

    #%%
    # Collect some info on the extracted dictionaries
    for idct, key_dct in enumerate(standard_info):
        dct = standard_info[key_dct]
        dict_len.append(len(dct))
        for key_sec in dct:
            if key_sec in sections_list:
                sections_count[key_sec] += 1
            else:
                sections_list.append(key_sec)
                sections_count[key_sec] = 1
            # if the section notes is present, get the filename because this was
            # a standard with missing URN (had the string '[URN]' instead)
            if key_sec == 'notes':
                all_urnmissing_files.append(which_file[20:30].replace(
                        '.','') + '/' + standard_files[idct])
                all_urnmissing_names.append(dct['Title'])
            
    texts.append('-'*60)

texts.append('Total numbers of extracted and failed files are {} and {}, respectively'.format(
        total_doc_extracted, total_doc_failed))

# now something about duplicates NOS
texts.append('Total numbers of unique and repeated URNs are {} and {}, respectively'.format(
        total_doc_unique - len(all_repeated_ref), len(all_repeated_ref)))

texts.append('Average number of repetitions for non-unique RNs is {}'.format(
        all_repeated_counts.mean()))
if all_repeated_counts.std()>0:
    high_rep_ref = np.where(all_repeated_counts>2)[0]
    texts.append('The number of RNs repeated more than two times is {}'.format(
            len(high_rep_ref)))
    texts.append('And these standards are: \n')
    for ii in high_rep_ref:
        texts.append(all_repeated_ref[ii])

# ADD example of extracted standard
texts.append('#'*80 + '\n')
texts.append('Example of extracted standard \n')

for key in dct:
    texts.append('\n Section name: {}. '.format(key))
    texts.append('Section content:')
    if isinstance(dct[key],list):
        for elem in dct[key]:
            texts.append(elem)
    else:
        texts.append(dct[key])

# finally, print a list of all failed NOS
# save everything on file and possibly print on screen too
if WRITEFILE:
    for text in texts:
        #print(text)
        fsave.write(text + '\n')


#plot histograms with count of section appearances across document
# (as a proportion of documents)
section_props = np.zeros((len(sections_count.keys())))
for ii,ikey in enumerate(sections_count):
    section_props[ii] = sections_count[ikey]/total_doc_extracted*100
    x = np.arange(ii+1) # this is a lazy and inefficient trick, yes

# sort them in descending order
IX = np.argsort(section_props)[::-1]
section_props = section_props[IX]
ticks = list(sections_count.keys())
ticks= [ticks[ii] for ii in IX]
fig, ax = plt.subplots(figsize = (8,4))
plt.bar(x, section_props)
plt.xticks(x, ticks, rotation = 'vertical')
plt.ylabel('% of NOS')
plt.xlabel('section')
plt.tight_layout()
plt.savefig(os.path.join(summary_dir, version_load + 'sections_counts.png'))
#plt.show()

# print the numbers too:
if WRITEFILE:
    fsave.write('\n')
    fsave.write('#'*80 + '\n')
    fsave.write('Counting how many times each section (i.e. performance criteria or glossary) appears in all the extracted NOS\n')
    fsave.write('\n')
    for ii,perc in enumerate(section_props):
        fsave.write('Section \'{}\' was extracted for {:.4f}% of the documents'.format(ticks[ii],perc))
        fsave.write('\n')

    fsave.write('\n')
    fsave.write('NOTE: The percentage of NOS with a \'note\' field (added during the \n'
                + 'extraction process) corresponds to those NOS that had \'[URN]\' \n'
            + 'or \'[Unique Reference Number]\' in the header instead of a proper \n' +
            'reference number. For these files I used the original URN field as their \n'
            + 'reference number.')
    fsave.write('\n')
    fsave.write('\n')

    fsave.write('#'*80 + '\n')
    fsave.write('All the standards that failed are: \n')
    for ifile in all_standard_failed:
        fsave.write(ifile)
        fsave.write('\n')

    fsave.close()
    
    # save the dictionary with the repeated URNs
    pd.DataFrame.from_dict(all_repeated_files, orient = 'index').to_csv(
            summary_dir + '/' + version_load + 'repeated_NOS.csv')
    
    # save the dictionary of files with missing URNs in their header
    #pd.DataFrame(all_urnmissing_files).to_csv(summary_dir + '/missing_urn_NOS.csv')
    pd.DataFrame({'filename of missing URN': all_urnmissing_files, 
                  'standard name': all_urnmissing_names}).to_csv(
                        summary_dir + '/' + version_load + 'missing_urn_NOS.csv')
