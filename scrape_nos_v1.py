#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 08:24:23 2019

@author: jdjumalieva

ORIGINAL VERSION of scrape_nos.py: please see "scrape_nos.py" for information.
Better not to use this one, it is kept mostly for reference.

"""

import os
from collections import defaultdict
import textract
import zipfile
import pickle
import shutil
import argparse
import logging
import sys
import time
import traceback

def find_indices(tables_local, all_possible_headers_local, \
    all_storage_names_local, ordered= False):
    # Note: this checks that a specific section is in the tables
    # If we know that all sections should be in there, then 
    # len(all_possible_headers) = len(all_headers_local)
    ind_start_local = {}
    ind_end_local = {}
    N_local = len(tables_local)
    all_headers_local = []
    all_storage_local = []
    all_indices_tmp_local = []
    # divide the nested tables from environment to environment
    ih_local = 0
    # find where each header is
    for ih2_local,head_local in enumerate(all_possible_headers_local):
        tmp = [ii for ii in range(N_local) if head_local in tables_local[ii]]
        if len(tmp):
            ind_start_local[head_local] = tmp[0]
            all_indices_tmp_local.append(tmp[0])
            all_headers_local.append(head_local)
            all_storage_local.append(all_storage_names_local[ih2_local])
            if ordered:
                if ih_local>0:
                    ind_end_local[head_local] = ind_start_local[all_headers_local[ih-1]]
                else:
                    ind_end_local[head_local] = N_local
                ih_local += 1
    if not ordered:
        # If the sections are not ordered sort the indices in ascending order so that 
        # you get the page order for this document
        #index_order = sorted(range(len(all_indices_tmp)), key= lambda k: all_indices_tmp[k])
        all_indices_tmp_local.sort()
        all_indices_tmp_local.append(N_local)
        for head_local in all_headers_local:
            # get starting index for this header
            val = ind_start_local[head_local]
            # find the next index
            next_val = all_indices_tmp[all_indices_tmp.index(val)+1]
            ind_end_local[head_local] = next_val
    return ind_start_local, ind_end_local, all_headers_local, all_storage

###################################################
# Main script starting
###################################################


t0start = time.time()

#oldstdout = sys.stdout

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data-dir', default = '../../data/NOSdata',
                    help='Main directory where to find all data. ' \
                    + 'Use question marks (?) instead of spaces if needed')
parser.add_argument('--input-dir', default = 'sample_zip.zip',
                    help='Zipped folder to extract data from. ' \
                    + 'Use question marks (?) instead of spaces, if needed. ' \
                    + 'If it is a pdf/zip file, explicitely add the extension at the end')
parser.add_argument('--save-dir', default = '../../data/NOSdata',
                    help='Main directory where to save extracted data. ' \
                    + 'Use question marks (?) instead of spaces, if needed. ')
parser.add_argument('--keep-unzip', action = 'store_true',
                    help = 'keep files that have been unzipped. ')

args = parser.parse_args()

data_dir = args.data_dir.replace('?',' ') #'../../data/NOSdata'
input_dir = args.input_dir.replace('?', ' ') #'sample_zip.zip'
REMOVE = not args.keep_unzip # default is to delete them

save_dir = args.save_dir.replace('?',' ') #'../../data/NOSdata'
#stdoutfile = os.path.join(data_dir,'log_output_' + os.path.splitext(input_dir)[0] + '.py').replace(' ','_')
#sys.stdout = open(stdoutfile, 'w')

standard_info = defaultdict(dict) #collect all the info structure
standard_ids = [] #collect all the IDs (i.e. the dictionary keys)
standard_failed = [] #collect those that failed

divider= '-'*95 + '\n'
print(divider)
print('Start scrape_nos with arguments: ')
for k in vars(args):
    print('{}: {}'.format(k,vars(args)[k]))
print(divider)

# Prep the zip folder #########################################################
if input_dir.endswith('.pdf'):
    file_list = [input_dir]
    unzipped_dir = data_dir
    # If it is a single file, do not delete it
    REMOVE = False

else:
    if input_dir.endswith('.zip'):
        print('Unzipping the required directory')
        t0 = time.time()
        with zipfile.ZipFile(os.path.join(data_dir,input_dir), mode = 'r') as zip_ref:
            unzipped_dir = os.path.join(data_dir,input_dir[0:-4]) #remove zip extension
            zip_ref.extractall(data_dir)
        print('Done. Time spent unzipping all files is {:4f} s'.format(time.time()-t0))
    else:
        unzipped_dir = os.path.join(data_dir, input_dir)
        # in this case, do not remove the files
        REMOVE = False
    #OBS: This assumes that the zipped folder contains another folder with THE SAME NAME!
    
    # Files should have been extracted in the directory data_dir+input_dir
    file_list = os.listdir(unzipped_dir)
        
        
    # only keep the files with pdf extension
    file_list = [ifile for ifile in file_list if ifile.endswith('.pdf')]


# Process each file in the list ###############################################
print('Overall number of files to extract is {}.'.format(len(file_list)))
print(divider)
t0 = time.time()
for nbfile,ifile in enumerate(file_list):
    print(divider)
    print('Start extract info from file (nb): {} ({})\n'.format(ifile,nbfile))
    # wrap everything in a try/except:
    try:
        #Prep the pdf##################################################################
        #Convert pdf to bytes
        #pdftext = textract.process(os.path.join(input_dir, 'New NOS 1/Carry-out-fresh-produce-handling-and-quality-IMPPP128.pdf'))
        #Address-enquiries-relating-to-financial-crime-from-those-with-empowered-authority--FSPCFC15.pdf')) #'sample_standard.pdf'))
        pdftext = textract.process(os.path.join(unzipped_dir, ifile))
        #Convert bytes to string
        rawtext = pdftext.decode('utf-8')
        
        #Get standard reference id
        standard_ref = rawtext.split('\n')[0]
            
        #Split content by standard reference id (this should get us standard pages)
        tables = rawtext.split(standard_ref)
        rawtext = None

        #Split by new lines and remove empty elements
        nested_tables = [elem.split('\n') for elem in tables if len(elem)]
        tables = None
        nested_tables2 = [[subelem for subelem in elem if len(subelem)] \
                           for elem in nested_tables]
        nested_tables = None

        print('Got to creating all nested tables')
        # get the standard name (it can be split across multiple rows)
        # the first nested table is the overview: everything before the word 
        # "overview" is the standard name
        # however, keep the line split because we need to remove its other occurences
        find_overview = nested_tables2[0].index('Overview')
        num_lines_name = find_overview
        standard_name = nested_tables2[0][0]
        standard_name_split1 = [standard_name]
        for ii in range(1,num_lines_name):
            standard_name = standard_name + ' ' + nested_tables2[0][ii]
            standard_name_split1.append(nested_tables2[0][ii:ii+1])

        # the second nested table is the footer of the first page
        # this footer contains the standard's name again, but in a different font, so
        # the name could be split again, but differently
        standard_name_split2 = [nested_tables2[1][0]]
        ii = 1
        # keep adding until you find the page number, i.e. 1
        # TODO: the following doesn't work for the Insert-and-remove-a-catheter-o-SFHCI.
        # The rawtext is split in a very weird way! Need to change
        while nested_tables2[1][ii]!='1':
            print(nested_tables2[1][ii])
            standard_name_split2.append(nested_tables2[1][ii:ii+1])
            ii += 1
        print('Got to storing the split names from the header and the footer. ', ii)
        #%%
        # Section search ##############################################################
        # Find the sections in the document + start and end "pages"
        all_possible_headers = ['Overview','Performance criteria','Knowledge and', \
                                'Scope/range','Glossary','External Links','Developed by'][::-1]
        all_storage_names = ['standard_overview','standard_perf','standard_know', \
                             'standard_scope','standard_glossary','ext_links','standard_others'][::-1]
        # does glossary come before or after Scope/range?
        
        indices_start, indices_end, all_headers, all_storage = find_indices(nested_tables2, all_possible_headers, \
    all_storage_names, ordered= False)
        N = len(nested_tables2)
        # TODO: keep in mind that the above function won't work is multiple sections are on the same page. Indeed if 
        # there are two same values then we'll have next_val = val, so for that section the start and end index will be  
        # the same
        print('Got the start and end indices for all sessions')

        # collect all the field names that we want to ignore
        all_to_remove = all_headers + [standard_name] + standard_name_split1 + standard_name_split2 \
            + ['You must be able to:'] + ['understanding'] + ['You need to know and'] + ['understand:'] + ['']
        # NOTE: This DOES NOT eliminate the page numbers, this happens later
        
        # do the same for the metadata
        search_meta_data = ['Version Number', 'Date Approved', 'Indicative Review', 'Validity', \
                         'Status', 'Originating', 'Original URN', 'Relevant', 'Suite', 'Keywords', 'Key words'][::-1]
        all_meta_data = search_meta_data + ['Date', 'Organisation', 'Occupations']
        name_meta_data = ['version_num','date_approved','indic_review_date','validity',\
                          'status','orig_org','orig_urn','relev_occup','suite','keywords','keywords'][::-1]
        
        # TODO: Check the orig_urn for "The carry-out-fresh-produce-handling-and-quality"
        # It splits the K out of the URN = ref. No idea how to avoid this
        
        #%%
        # Fill in the info dictionary ################################################
        standard_info[standard_ref]['standard_name'] = standard_name
        
        for ih,head in enumerate(all_headers):
            tmp_table = nested_tables2[indices_start[head]]
            for ii in range(indices_start[head]+1,indices_end[head]):
                tmp_table = tmp_table + nested_tables2[ii]
            if head == 'Developed by':
                meta_info = [elem for elem in tmp_table if elem not in all_to_remove]#[0:-2]
                # first line should be who wrote the standard
                standard_info[standard_ref]['developed_by']= meta_info[0] 
                meta_info= meta_info[1:]
                #look for the sub-indices
                sub_indices_start,  sub_indices_end, all_metas, all_storage_meta = find_indices(meta_info, all_meta_data, name_meta_data, ordered= True)
                subN = len(meta_info)
                for ih_meta, meta in enumerate(all_metas):
                # divide the nested tables from environment to environment
                    tmp_table2 = meta_info[sub_indices_start[meta]:sub_indices_start[meta]+1]
                    # need to slice differently otherwise it returns strings rather than lists
                    for jj in range(sub_indices_start[meta]+1,sub_indices_end[meta]):
                        tmp_table2 = tmp_table2 + meta_info[jj:jj+1]
                    standard_info[standard_ref][all_storage_meta[ih_meta]] = [elem for elem in tmp_table2 if elem not in                                     all_meta_data]
            elif head in ['Skills','Links to other NOS','External Links']:
                # The problem with this is that it doesn't always start with the skills
                pass
            else:
                standard_info[standard_ref][all_storage[ih]] = [elem for elem in tmp_table if elem not in \
                                 all_to_remove]#[0:-2]
                    # the last two lines should always be page number + empty line.
                    # TODO: I'm not sure of the above, so I'm keeping those lines for now

        print('Extracted info for all headers')
        
        # the last page could have been split in two if the stardard_name also appears
        # as the original urn: if this has happened, add it
        if indices_end['Developed by'] - indices_start['Developed by']> 2:
            standard_info[standard_ref]['orig_urn'] += [standard_ref]
        standard_ids.append(standard_ref)

        print('Added the original URN if needed')

        # save all as pickled file
        save_file = os.path.join(save_dir,'extracted_standards_' + os.path.splitext(input_dir)[0].replace('/','-') + '.pickle')
        #save_file = save_file.replace(' ','_')#.replace('/','-')
        print(save_file)
        with open(save_file,'wb') as f:
            pickle.dump((standard_info,standard_ids,standard_failed),f)

        print('Done. All went well.')

    except Exception: # as ex:
        print('#'*30 + '\n')
        print('Warning! ', sys.exc_info()[0], ' exception occured. File (nb) {} ({}) had a problem. \n'.format(ifile,nbfile))
        #logging.exception('Caught an error with file ' + ifile + '\n')
        #tb = sys.exc_info()[2]
        traceback.print_exc(file=sys.stdout)
        standard_failed.append(os.path.join(unzipped_dir,ifile))

    if (nbfile+1)%100 == 0:
        print('Time spent on the last 100 files is {:.4f} s'.format(time.time()-t0))
        t0 = time.time()        
        
print(divider)
# if requested, remove entire unzipped folder
if REMOVE:
    shutil.rmtree(unzipped_dir)
    
#sys.stdout= oldstdout

print('Total time to extract and process {} files is {:4f} s'.format(len(file_list),time.time() - t0start))

