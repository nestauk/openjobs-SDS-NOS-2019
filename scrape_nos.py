#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 1 11:24:23 2019


Description:

    This script is to extract relevant information from National Occupational
    Standards (NOS). For example, it extracts info like standard ref, document
    overview, performance criteria, version number, etc.

    Inputs: either a zip folder, a normal directory or a pdf file. If the input
    is a sip folder, the script will delete the extracted files by default

    Output: it saves all the extracted standards from the input folder as a
    dictionary, where the key is the standard reference number. It also saves
    a list of files for which the extraction failed; all the reference numbers
    and the file names of extracted NOS.

    Other arguments:
        --save-dir: where to save the final .pickle file with the outputs
        --input-dir: the file(s) to extrac
        --data-dir: the directory where to find the file(s) to extract. It's
            kept separate from the input dir because often the directory stay
            the same, while the specific input file changes
        --keep-unzip: if added, it does NOT delete the unzipped files.

    OBS: when inputing files, please substitute spaces in file names with
    question marks (?)

    Typical call from command line is:
        python scrape_nos.py --input-dir sample_standard.pdf > log_sample_standard.txt

    Alternative, it can be called multiple times with different arguments
    using the script "batch_scrape_nos.py"


"""

# TODO: need to change directory in which to extract files - it keeps
# taking up space in my Google Drive even if I then delete the files
import os
from collections import defaultdict
import textract
import zipfile
import pickle
import shutil
import argparse
#import logging
import sys
import time
import traceback
import copy

def find_indices(tables_local, all_possible_headers_local, \
    all_storage_names_local, ordered= False, lower = False):
    '''
    This function finds where in the document we can find specific section headers
    (listed in all_possible_headers)
    It does not assume that all the headers will be found in each document

    It returns a list of the headers that were found, together with their names
    (names are a subset of all_storage_names) and with their start and end
    indices. The latter are stored in dictionaries with the headers as the keys
    '''
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
        # find pages that mention this specific header
        if lower and head_local != 'Suite':
            # here each element of the tables is a single row of the NOS
            # thus we can use "=="
            tmp = [ii for ii in range(N_local) if head_local.lower() == tables_local[ii].lower()]
        else:
            # here each element of the tables is an entire page of the NOS
            # thus we use "in"
            tmp = [ii for ii in range(N_local) if head_local in tables_local[ii]]
        if len(tmp):
            ind_start_local[head_local] = tmp[0] # get the first occurence (is this robust?)
            all_indices_tmp_local.append(tmp[0])
            all_headers_local.append(head_local)
            all_storage_local.append(all_storage_names_local[ih2_local])
            if ordered:
                if ih_local>0:
                    ind_end_local[head_local] = ind_start_local[all_headers_local[ih_local-1]]
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
            next_val = all_indices_tmp_local[all_indices_tmp_local.index(val)+1]
            ind_end_local[head_local] = next_val
    return ind_start_local, ind_end_local, all_headers_local, all_storage_local


def check_unique_ind(head_local, indices_start, indices_end):
    '''
    This function checks whether two or more headers are on the same ``section''
    (it allows for page brakes)
    '''
    indices_start_local = copy.deepcopy(indices_start)
    indices_end_local = copy.deepcopy(indices_end)
    # Need to check if there is a shared start/end index or if another header starts before this one finishes
    # initially assume that it's unique
    flag_unique = True
    # not check for all the conditions that would make it not unique
    target_ind_start = indices_start_local[head_local]
    target_ind_end = indices_end_local[head_local]
    flag = indices_start_local.pop(head_local,0)
    if not flag:
        print('Something is wrong with {}'.format(head_local))
    flag = indices_end_local.pop(head_local,0)
    if not flag:
        print('Something is wrong with {}'.format(head_local))
    for key_local in indices_start_local:
        # loop over the other indices and check for any repetition
        if indices_start_local[key_local] == target_ind_start:
            flag_unique = False
            break # one condition is enough to stop
        elif indices_start_local[key_local] > target_ind_start and indices_start_local[key_local] < target_ind_end:
            # for normal behaviour, if another header starts after this one,
            # it would start after this one has ended
            flag_unique = False
            break
    if flag_unique:
        # no problem found with the start indices, check the end indices
        for key_local in indices_end_local:
            if indices_end_local[key_local] == target_ind_end:
                flag_unique = False
                break # one condition is enough to stop
    return flag_unique


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
parser.add_argument('--save-dir', default = '../../results/NOS/extracted',
                    help='Main directory where to save extracted data. ' \
                    + 'Use question marks (?) instead of spaces, if needed. ')
parser.add_argument('--keep-unzip', action = 'store_true',
                    help = 'keep files that have been unzipped. ')

args = parser.parse_args()

data_dir = args.data_dir.replace('?',' ') #'../../data/NOSdata'
input_dir = args.input_dir.replace('?', ' ') #'sample_zip.zip'
REMOVE = False #not args.keep_unzip # default is to delete them

save_dir = args.save_dir.replace('?',' ') #'../../data/NOSdata'
#stdoutfile = os.path.join(data_dir,'log_output_' + os.path.splitext(input_dir)[0] + '.py').replace(' ','_')
#sys.stdout = open(stdoutfile, 'w')

standard_info = defaultdict(dict) #collect all the info structure
standard_ids = [] #collect all the IDs (i.e. the dictionary keys)
standard_failed = [] #collect those that failed
standard_files = []

divider= '-'*95 + '\n'
print(divider)
print('Start scrape_nos with arguments: ')
for k in vars(args):
    print('{}: {}'.format(k,vars(args)[k]))
print(divider)

### Define all the headers ####################################################
all_possible_headers = ['Overview','Performance criteria','Performance','Knowledge and', \
                        'Scope/range','Scope / range', 'Scope/range related', 'Scope related to', 'Glossary','External Links','Values','Behaviours', \
                        'Skills','Links to other NOS','Developed by'][::-1]
# only for the first three we are sure they start on a separate page
secondary_headers = ['Scope/range','Glossary','External Links','Values','Behaviours',\
                     'Skills','Links to other NOS']
all_storage_names = ['Overview','Performance_criteria', 'Performance_criteria', \
                     'Knowledge_and_understanding', \
                     'Scope_range','Scope_range','Scope_range','Scope_range', \
                     'Glossary','External_Links','Values','Behaviours',\
                     'Skills','Links_to_other_NOS','standard_others'][::-1]
# and the metadata in the last page
search_meta_data = ['Developed by', 'Version Number', 'Date Approved', 'Indicative Review', \
                    'Validity', \
                 'Status', 'Originating', 'Original URN', 'Relevant', \
                 'Suite', 'Keywords', 'Key words'][::-1]
all_meta_data = search_meta_data + ['Date', 'date', 'Organisation', 'organisation', \
                                    'Occupations', 'occupations']
# make it all lower case
all_meta_data = [ii.lower() for ii in all_meta_data]
name_meta_data = ['Developed_by', 'Version_number', 'Date_approved', \
                    'Indicative_review_date', 'Validity', \
                 'Status', 'Originating_organisation', 'Original_URN', \
                 'Relevant_occupations','Suite','Keywords','Keywords'][::-1]


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
        # if not a zip folder, do not remove the files
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
        standard_tmp = {}

        # add the file name
        standard_tmp['file_name'] = ifile

        #Prep the pdf##################################################################
        #Convert pdf to bytes
        pdftext = textract.process(os.path.join(unzipped_dir, ifile))
        #Convert bytes to string
        rawtext = pdftext.decode('utf-8')

        # Some NOS have this weird marker: "NOT PROTECTIVELY MARKED'
        rawtext = rawtext.replace('NOT PROTECTIVELY MARKED','')
        #Get standard reference id
        standard_ref = rawtext.split('\n')[0]

        # it might be that the first few lines are actually empty
        if not len(standard_ref):
            standard_ref = rawtext.split('\n')[1]
            if not len(standard_ref):
                standard_ref = rawtext.split('\n')[2]

        #Split content by standard reference id (this should get us standard pages)
        # It doesn't get the standard pages because the URN can be found in the text as well
        #tables = rawtext.split(standard_ref)
        # the solution might be to find the last page and split by page numbers
        last_page = 0
        y = [t for t in rawtext[-10:].split('\n') if t.isdigit()]
        last_page_str = ''
        #if len(y):
        #    for t in y:
        #        last_page_str += t
        #    last_page = int(last_page_str)
        for elem in rawtext[-10:].split('\n'):
            try:
                last_page = int(elem)
            except:
                pass
        #%%
        #last_page = 0
        if last_page:
            # Here we're still creating a list with ideally one page per list element
            # if it has found the last page, great
            # however, we need a more robust split string (if it's only the page number,
            # it can interfer with other numbers in the text)
            try:
                split_text = '\n\n{}\n\n' + rawtext.split('\n\n1\n\n')[1][:10]# + '\n{}\n'
                tables = []
                for ipage in range(1,last_page):
                    tmp = rawtext.split(split_text.format(ipage))
                    if len(tmp)>1:
                        tables.append(tmp[0]) # + split_text[:10])
                        rawtext = split_text[5:] + tmp[1]
                    else:
                        # if we are here, something went wrong in how the file was read
                        # try to split by the page number and hope that the same
                        # number does not appear in the body of the page as well
                        tmp = rawtext.split('\n{}\n'.format(ipage))
                        tables.append(tmp[0])
                        rawtext = tmp[1]
                        # add all the sections in case the same number appears later on
                        for ii in range(2,len(tmp)):
                            print(ii)
                            rawtext = rawtext + tmp[ii]
                # at the end append the last page
                tables.append(tmp[1][0:-2])
            except:
                # I did not consider that this might be happening after the
                # original rawtext has been modified already
                # reload the rawtext first
                rawtext = pdftext.decode('utf-8')
                # Some NOS have this weird marker: "NOT PROTECTIVELY MARKED'
                rawtext = rawtext.replace('NOT PROTECTIVELY MARKED','')
                # now split
                tables = rawtext.split(standard_ref)
                last_page = 0
        else:
            # if not, resort to the previous solution. The rest of the script
            # should work the same since the working principles don't change
            tables = rawtext.split(standard_ref)
        #%%
        #rawtext = None
        # sometimes the URN can appear on the second page, but not on the first
        # one - the first one has either a string with 'Unique Reference Number'
        # or one containing .docx
        # For those cases, try getting the URN from the last page after the split
        if '.docx' in standard_ref or 'Unique Ref' in standard_ref:
            standard_ref= tables[1].split('\n')[1].replace('\x0c','')

        #Split by new lines and remove empty elements
        nested_tables = [elem.split('\n') for elem in tables if len(elem)]
        #tables = None
        nested_tables2 = [[subelem for subelem in elem if len(subelem)] \
                           for elem in nested_tables]
        #nested_tables = None

        print('Got to creating all nested tables')
        # get the standard name (it can be split across multiple rows)
        # the first nested table is the overview: everything before the word
        # "overview" is the standard name
        # however, keep the line split because we need to remove its other occurences
        find_overview = nested_tables2[0].index('Overview')
        num_lines_name = find_overview
        # if I split by last page numbers, the first line is the standard ref
        start_for_name = 1 if last_page else 0
        standard_name = nested_tables2[0][start_for_name]
        standard_name_split1 = [standard_name]
        for ii in range(start_for_name+1,num_lines_name):
            standard_name = standard_name + ' ' + nested_tables2[0][ii]
            standard_name_split1.append(nested_tables2[0][ii])#:ii+1])

        # the second nested table is the footer of the first page
        # this footer contains the standard's name again, but in a different font, so
        # the name could be split again, but differently
        if last_page:
            # Not sure, but if I split by page number the footer should be the
            # last few lines of the first table (=first page) - want to store it to
            # then remove it from all other pages
            # I should start from the bottom and work up until I find the standard ref
            standard_name_split2 = [nested_tables2[0][-1]]
            ii = -2
            if standard_ref not in standard_name_split2[-1]:
                standard_name_split2.append(nested_tables2[0][ii])
                ii = ii-1
        else:
            # otherwise it's as before
            standard_name_split2 = [nested_tables2[1][0]]
            ii = 1
            # keep adding until you find the page number, i.e. 1
            # TODO: the following doesn't work for the New NOS 1/Insert-and-remove-a-catheter-o-SFHCI.
            # The rawtext is split in a very weird way! Need to change
            max_iter = len(nested_tables2[1])
            for ii in range(1,max_iter):
                if nested_tables2[1][ii]!='1':# and ii<max_iter:
                    print('For this DOC the page split was weird and the name was not where I thought it would be')
                    standard_name_split2.append(nested_tables2[1][ii])#:ii+1])
                    #ii += 1
                else:
                    # stop as soon as we get to the page number
                    break;
        print('Got to storing the split names from the header and the footer. ', ii)

        # Section search ##############################################################
        # Find the sections in the document + start and end "pages" (headers above)

        indices_start, indices_end, all_headers, all_storage = find_indices(nested_tables2, all_possible_headers, \
    all_storage_names, ordered= False, lower = False)
        N = len(nested_tables2)
        # TODO: keep in mind that the above function won't work is multiple sections are on the same page.
        # Indeed if there are two same values then we'll have next_val = val, so for that section the start and
        # end index will be the same.
        print('Got the start and end indices for all sessions')

        # collect all the field names that we want to ignore
        all_to_remove = all_headers + [standard_name] + standard_name_split1 + standard_name_split2 \
            + ['You must be able to:'] + ['understanding'] + ['You need to know and'] + ['understand:'] + [''] \
                + ['Additional Information'] + ['related to'] + ['performance'] + ['criteria'] \
                    + ['knowledge and'] + ['understand'] #last one not sure it's needed
        # NOTE: This DOES NOT eliminate the page numbers

        # do the same for the metadata (metadata defined above)
        # TODO: Check the orig_urn for "The New?NOS?1/Carry-out-fresh-produce-handling-and-quality-IMPPP128.pdf"
        # It splits the K out of the URN = ref. No idea how to avoid this

        # Fill in the info dictionary ################################################
        standard_tmp['Title'] = standard_name

        for ih,head in enumerate(all_headers):
            # accumulate all pages that refer to a specific header
            tmp_table = nested_tables2[indices_start[head]]
            for ii in range(indices_start[head]+1,indices_end[head]):
                tmp_table = tmp_table + nested_tables2[ii]
            # go case by case
            if head == 'Developed by':
                all_to_remove_tmp = [rem for rem in all_to_remove if rem != 'Developed by']
                meta_info = [elem for elem in tmp_table if elem not in all_to_remove_tmp]
                #meta_info_keep = meta_info
                ## first or second line should be who wrote the standard
                #dev_line = 1 if last_page else 0
                #standard_tmp['developed_by']= meta_info[dev_line]
                #meta_info= meta_info[dev_line+1:]
                # look for the sub-indices
                sub_indices_start,  sub_indices_end, all_metas, all_storage_meta = \
                find_indices(meta_info, search_meta_data, name_meta_data, ordered= True, lower = True)
                subN = len(meta_info)

                # There might be some text that preceded the developed by field
                # capture it as extra text. Note: there has to be at least 4 lines
                # of extra text for us to want to store it
                tmp_min = min([sub_indices_start[k] for k in sub_indices_start])
                if tmp_min > 2:
                    standard_tmp['extra_meta_info'] = meta_info[0:tmp_min]
                for ih_meta, meta in enumerate(all_metas):
                # divide the nested tables from environment to environment
                    tmp_table2 = meta_info[sub_indices_start[meta]:sub_indices_start[meta]+1]
                    # need to slice differently otherwise it returns strings rather than lists
                    for jj in range(sub_indices_start[meta]+1,sub_indices_end[meta]):
                        tmp_table2 = tmp_table2 + meta_info[jj:jj+1]
                    standard_tmp[all_storage_meta[ih_meta]] = \
                            [elem for elem in tmp_table2 if elem.lower() not in all_meta_data]
            elif head in secondary_headers:
                # check if it appears on a page by itself
                flag_unique = check_unique_ind(head, indices_start, indices_end)
                if flag_unique:
                    # in this case, proceed as usual
                    standard_tmp[all_storage[ih]] = [elem for elem in tmp_table if elem not in \
                                 all_to_remove]#[0:-2]
                else:
                    # it doesn't appear by itself. It means it could be anywhere in the page, together with many
                    # other secondary headers
                    # parse the table line by line looking for the headers
                    found_head = 0
                    # start by assuming this header is the last in the page, or you'll get an error
                    end_in_pages = len(tmp_table)
                    for ielem, elem in enumerate(tmp_table):
                        if not found_head:
                            # look for the header if it has not yet been found
                            if elem == head:
                                found_head = 1
                                start_in_pages = ielem
                        else:
                            # once you have found the header, look for the next one
                            if elem in secondary_headers:
                                end_in_pages = ielem
                                break #stop the cycle, the rest of the table is irrelevant
                    # select the relevant part of the table
                    tmp_table2 = tmp_table[start_in_pages:end_in_pages]
                    # remove the rest of the unwanted lines
                    standard_tmp[all_storage[ih]] = [elem for elem in tmp_table2 if elem not in \
                        all_to_remove]

            else:
                # this is for those headers that we know are on separate pages
                standard_tmp[all_storage[ih]] = [elem for elem in tmp_table if elem not in \
                                 all_to_remove]

        print('Extracted info for all headers')

        # the last page could have been split in two if the stardard_name also appears
        # as the original urn: if this has happened, add it
        if indices_end['Developed by'] - indices_start['Developed by']> 2:
            standard_tmp['Original_URN'] += [standard_ref]

        # Some files have a missing URN name in the footer. For those, use the
        # original urn, but leave a note about it
        #if standard_ref in ['[URN]', '[Unique Reference Number]']:
        if 'URN' in standard_ref or 'Unique Ref' in standard_ref:
            standard_ref = standard_tmp['Original_URN'][0]
            standard_tmp['notes'] = 'Original URN was used as the standard ref'

        standard_ids.append(standard_ref)
        standard_files.append(ifile)
        print('Added the original URN if needed')

        standard_tmp['URN'] = standard_ref

        # if the standard_ref is already in there, save it as version 2
        if standard_ref not in standard_info:
            standard_info[standard_ref] = standard_tmp
        else:
            standard_info[standard_ref + '_v2'] = standard_tmp

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

# save all as pickled file
prefix = 'new_'
save_file = os.path.join(save_dir, prefix + 'extracted_standards_' + os.path.splitext(os.path.basename(input_dir))[0] + '.pickle')
#save_file = os.path.join(save_dir,'extracted_standards_' + os.path.splitext(input_dir)[0].replace('/','-') + '.pickle')
#save_file = save_file.replace(' ','_')#.replace('/','-')
print(save_file)
with open(save_file,'wb') as f:
    pickle.dump((standard_info,standard_ids,standard_failed,standard_files),f)

print(divider)
# if requested, remove entire unzipped folder
if REMOVE:
    shutil.rmtree(unzipped_dir)

#sys.stdout= oldstdout

N1 = len(standard_failed)
N2 = len(file_list)
print('Number of failed-to-extract files is {} ({:4f}% of {} files in total)'.format(N1, N1/N2*100, N2))
print('Total time to extract and process {} files is {:4f} s'.format(N2,time.time() - t0start))


'''
### Parking lot

            elif head in ['Skills','Links to other NOS','External Links']:
                # The problem with this is that it doesn't always start with the skills
                # TODO!!!
                # get which ones are actually is the NOS
                flag = '1' if 'Skills' in all_headers else '0'
                flag += '1' if 'Links to other NOS' in all_headers else '0'
                flag += '1' if 'External Links' in all_headers else '0'
                ind_start_all = []
                ind_end_all = []
                keys_list = ['Skills','Links to other NOS','External Links']
                # get all the start and end indices, then find the earliest start and latest end (they can be split on multiple pages)
                for ikey,key_local in enumerate(keys_list):
                    if flag[ikey]=='1':
                        ind_start_all.append(indices_start[key_local])
                        ind_end_all.append(indices_end[key_local])
                ind_start_skills = min(ind_start_all)
                ind_end_skills = max(ind_end_all)
                # get the pages of interest:

                # now go case by case in terms of what you're looking for
                if flag == '111':
                pass


'''
