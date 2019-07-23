#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:29:44 2019

@author: stefgarasto
"""


import os
import pickle
import re
import matplotlib.pyplot as plt
import pandas as pd

####### Definitions
def unique(list1):

    # intilize a null list
    unique_list = []
    repeated_elems = []
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
        else:
            repeated_elems.append(x)
    return unique_list, repeated_elems

######## Main code

data_dir = '../../results/NOS/extracted'
which_files = ['extracted_standards_New NOS 1.pickle', 'extracted_standards_New NOS 2.pickle']
for ii in range(1,23):
    which_files.append('extracted_standards_Old NOS {}.pickle'.format(ii))
#which_file = 'extracted_standards_Old NOS 22.pickle'

repeated_refs = []
tot_count = 0
# load all the extracted NOS and concatenate them
for ii,which_file in enumerate(which_files):
    print(which_file)
    with open(os.path.join(data_dir, 'v2_' + which_file),'rb') as f:
        standard_info_partial, standard_ref_partial, _, _ = pickle.load(f)
    tot_count += len(standard_ref_partial)
    # quick check that the list of keys in the standard dictionary is the
    # same as the URNs in the standard_ref list
    tmp = [k.replace('_v2','').replace('_v3','').replace('_v4','') for k in standard_info_partial.keys()]
    if not unique(standard_ref_partial)[0] == unique(tmp)[0]:
        print('Dictionary keys and standard refs do not correspond. Something is wrong.')
    if ii == 0:
        standard_info= standard_info_partial
        standard_ref = standard_ref_partial
    else:
        # check if there is any ref that is already in the full list
        ref_intersection= list(set(standard_ref_partial).intersection(set(standard_ref)))
        if len(ref_intersection):
            #keep track of the repeated ones and add _v followed by the next 
            # available number to their dict keys
            repeated_refs = repeated_refs + ref_intersection
            for ref in ref_intersection:
                new_version = 2
                while ref + '_v{}'.format(new_version) in list(standard_info_partial.keys()):
                    new_version+=1
                    #print(new_version)
                standard_info_partial[ref + '_v{}'.format(new_version)
                    ] = standard_info_partial.pop(ref)
        standard_info.update(standard_info_partial)
        standard_ref = standard_ref + standard_ref_partial


standard_info_partial = None
standard_ref_partial = None

# check if it's fine to turn some of the fields into string
# Specifically, these fields should be strings: URN, Title, Developed_by, Date_approved
# Indicative review date, Validity, Status, Originating organisation, Suite
# OBS:I would keep the Original URN as a list, since it can contain multiple one
# How about keywords?
# Also, version number should be an integer
for idct, key_dct in enumerate(standard_info):
    dct = standard_info[key_dct]
    for key in ['Developed_by','Date_approved', 'Indicative_review_date', 
                'Originating_organisation', 'Status','Suite', 'Validity','Version_number']:
        if key in dct.keys():
            if len(dct[key]):
                tmp = dct[key][0]
                for ii in range(1,len(dct[key])):
                    tmp += dct[key][ii]
                dct[key] = tmp
                if key=='Version_number':
                    # sometimes, the Version number contains spurious characters, not only digit
                    # so I need to dig out whether there is a digit in it and take it out
                    # yes: sometimes this can return the wrong value, because it will stop at the
                    # first digit it encounters
                    s = dct['Version_number']
                    re_result = re.search('\d+',s)
                    if re_result:
                        dct['Version_number'] = int(s[re_result.span()[0]:re_result.span()[1]])
                    else:
                        dct['Version_number'] = -1
            else:
                if key=='Version_number':
                    dct['Version_number'] = -1
                else:
                    dct[key] = 'empty'
    

    #int(float(dct['Version_number']))

#%%
# Collect some info on the extracted dictionaries
dict_len = []
sections_list = []
sections_count = {}
for idct, key_dct in enumerate(standard_info):
    dct = standard_info[key_dct]
    dict_len.append(len(dct))
    for key_sec in dct:
        if key_sec in sections_list:
            sections_count[key_sec] += 1
        else:
            sections_list.append(key_sec)
            sections_count[key_sec] = 1


#%% Turn the nested dictionary into a dataframe
nos_data = pd.DataFrame.from_dict(standard_info, orient = 'index')

#%%
