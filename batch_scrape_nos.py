#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 12:11:41 2019

@author: stef

Wrapper to call the script "scrape_nos.py" multiple times with different command
line arguments.
See scape_nos.py for more info about each argument.


"""

import os
import time
import pickle
from collections import defaultdict
import copy

t00 = time.time()
output_dir = '../../results/NOS/extracted'
# concatenate all NOS folders

list_to_process= ['New NOS 1', 'New NOS 2','Old NOS 1','Old NOS 2','Old NOS 3','Old NOS 4']
for inos in range(5,23):
    #if inos in [5,6,7,9,10,12,19,21]:
    list_to_process.append('Old NOS {}'.format(inos))
    #else:
    #    list_to_process.append('Old NOS {}.zip'.format(inos))
print(list_to_process)

for elem in list_to_process[23:24]:
    prefix = 'new_'
    t0 = time.time()
    no_space_elem = elem.replace(' ','?')
    underscore_elem = os.path.splitext(elem.replace(' ','_'))[0]
    output_file = os.path.join(output_dir, prefix + 'log_extract_' + os.path.basename(underscore_elem))
    print('#'*40)
    print('Processing command:')
    cmd = 'python scrape_nos.py --input-dir {} > {}.txt'.format(no_space_elem,output_file)
    print(cmd)
    os.system(cmd)
    print('All done. Time elapsed: {:4f}'.format(time.time() - t0))

print('Time to process them all: {:4f} minutes'.format((time.time() - t00)/60.0))

# now refine them - you can overwrite the original file
print('Now removing some more spurious elements \n')

data_dir = '../../results/NOS/extracted'
#set which NOS dictionaries to analyse
which_files = ['extracted_standards_New NOS 1.pickle', 'extracted_standards_New NOS 2.pickle']
for ii in range(1,23):
    which_files.append('extracted_standards_Old NOS {}.pickle'.format(ii))
#which_file = 'extracted_standards_Old NOS 22.pickle'


for which_file in which_files[23:24]:
    print('Working on: ', which_file)

    REMOVE_REF = True
    if REMOVE_REF:
        # prefix of the NOS dictionary file to modify
        version_from = 'new_'
        # prefix for the new NOS dictionary file to save
        version_to = 'new_'
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

