#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:40:08 2019

@author: stefgarasto

This script is to refine the extraction of NOS files.

All these operations could be added to the main script - if/when they are added
they will be removed from here.

However, having a script like this allows for post-extraction modifications, as
well as being able to modify the extracted dictionaries without risks to the
original extraction code

For

The first thing we can do, by setting REMOVE_REF = True
is to eliminate the standard ref when it appears
in the beginning of a dictionary element.
This is a legacy of splitting by page number: the standard ref in the header
and in the footer is not removed



"""

import os
import pickle
from collections import defaultdict
import copy

data_dir = '../../results/NOS/extracted'
#set which NOS dictionaries to analyse
which_files = ['extracted_standards_New NOS 1.pickle', 'extracted_standards_New NOS 2.pickle']
for ii in range(1,23):
    which_files.append('extracted_standards_Old NOS {}.pickle'.format(ii))
#which_file = 'extracted_standards_Old NOS 22.pickle'


for which_file in which_files:
    print('Working on: ', which_file)

    REMOVE_REF = True
    if REMOVE_REF:
        # prefix of the NOS dictionary file to modify
        version_from = ''
        # prefix for the new NOS dictionary file to save
        version_to = 'v2_'
        with open(os.path.join(data_dir, version_from + which_file),'rb') as f:
            standard_info, standard_ref, standard_failed, standard_files = pickle.load(f)
            key_list = ['Overview', 'Knowledge_and_understanding' , 'Performance_criteria']
            # eliminate the standard ref when it appears in the beginning
            # it is a legacy of splitting by page number: the standard ref in the header
            # and in the footer is not removed
            standard_info2 = defaultdict(dict)
            for key_dct in standard_info:
                dct = copy.deepcopy(standard_info[key_dct])
                dct_ref = ['\x0c' + key_dct] + ['\x0c[URN]'] + ['\x0c[Unique Reference Number]']
                dct_ref = dct_ref + [key_dct] + ['[URN]'] + ['[Unique Reference Number]']
                #print(dct_ref)
                for key_sec in dct:
                    if isinstance(dct[key_sec],list):
                        dct[key_sec] = [elem for elem in dct[key_sec] if elem not in dct_ref]
                standard_info2[key_dct] = dct
            with open(os.path.join(data_dir, version_to + which_file),'wb') as f:
                pickle.dump((standard_info2, standard_ref, standard_failed, standard_files), f)
