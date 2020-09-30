'''
This script saves the NOS extracted from pdf as json files.
It follows the JSON schema from ActiveIS:
    
"URN": string,
  "Title": string,
  "Overview": list of string,
  "Performance_criteria": list of string,
  "Knowledge_and_understanding": list of string,
  "Scope_range": list of string,
  "Scope_range_related_to_performance_criteria": list of string,
  "Scope_range_related_to_knowledge_and_understanding": list of string,
  "Values": list of string,
  "Behaviours": list of string,
  "Skills": list of string,
  "Glossary": list of string,
  "Links_to_other_NOS": list of string,
  "External_Links": list of string,
  "Developed_by": string,
  "Version_number": numeric string,
  "Date_approved": "DAY MONTH YEAR",
  "Indicative_review_date": "DAY MONTH YEAR",
  "Validity": string,
  "Status": string,
  "Originating_organisation": string,
  "Original_URN": string,
  "Relevant_occupations": list of strings,
  "Suite": list of strings,
  "Keywords": list of strings,
  "SOC_Code": ;-delimited string,
  "NOSCategory": string
  
  
The df_nos columns start as:
URN <class 'str'>
NOS Title <class 'str'>
Overview <class 'list'>
Knowledge_and_understanding <class 'list'>
Performance_criteria <class 'list'>
Scope_range <class 'float'>
Glossary <class 'float'>
Behaviours <class 'float'>
Skills <class 'float'>
Values <class 'float'>
Links_to_other_NOS <class 'float'>
External_Links <class 'float'>
Developed By <class 'str'>
Version_number <class 'numpy.float64'>
Date_approved <class 'str'>
Indicative Review Date <class 'pandas._libs.tslibs.timestamps.Timestamp'>
Validity <class 'str'>
Status <class 'str'>
Originating_organisation <class 'str'>
Original URN <class 'str'>
Occupations <class 'str'>
All_suites <class 'list'>
Keywords <class 'str'>
Clean SOC Code <class 'list'>
NOSCategory <class 'str'>

Relevant_occupations <class 'list'>
notes <class 'float'>
empty <class 'numpy.bool_'>
extra_meta_info <class 'float'>
Created <class 'pandas._libs.tslibs.timestamps.Timestamp'>
Path <class 'str'>
pruned <class 'list'>
clean_full_text <class 'str'>
tagged_tokens <class 'list'>
full_text <class 'str'>
Date_approved_year <class 'numpy.float64'>
Title <class 'str'>
NOS Document Status <class 'str'>
Suite <class 'str'>
SuiteMetadata <class 'str'>
OccupationsMetadata <class 'str'>
One_suite <class 'str'>
Clean Ind Review Year <class 'numpy.float64'>
'''



#%matplotlib inline
import matplotlib.pyplot as plt


import os
import itertools
import json
import numpy as np
import pandas as pd
import pickle
from collections import Counter, OrderedDict
import time

stopwords= []
def get_keywords_list(x):
    #all_keywords = []
    #x = df_line['Keywords']
    if isinstance(x, list):
        # I think that ik can be a collection of words separated by ";" or ","
        ik_elems0 = ' '.join([elem for elem in x if not elem.isdigit()])
        ik_elems0 = ik_elems0.replace('-', ' ').replace(':','').replace(',',';')
        ik_elems0 = ik_elems0.replace('(','').replace(')','')
        ik_elems0 = ik_elems0.split(';')
        # remove extra spaces and make lowercase
        ik_elems0 = [elem.strip().lower() for elem in ik_elems0]
        ik_elems0 = [elem for elem in ik_elems0 if len(elem)]
        ik_elems = []
        for ik_elem in ik_elems0:
            ik_elem0 = ' '.join([elem for elem in ik_elem.split() if
                  (not elem.isdigit()) & (elem not in stopwords) & len(elem)])
            if len(ik_elem0):
                ik_elems.append(ik_elem0)
                    #print(ik_elems)
        return [elem.strip() for elem in ik_elems if len(elem)]
    elif isinstance(x,str):
        #ik_elems = re.findall(r"[\w']+", df.loc[ix].replace('-',''))
        ik_elems = x.replace('-', ' ').replace(',',';')
        ik_elems = ik_elems.replace('(','').replace(')','').split(';')
        if len(ik_elems)==1:
            # lacking proper separators - have to use spaces
            ik_elems = ik_elems[0].split()
        # remove extra spaces
        ik_elems = [elem.strip().lower().replace('\n','') for elem in ik_elems]
        # remove digits
        ik_elems = [elem for elem in ik_elems if not elem.isdigit()]
        ik_elems = [elem for elem in ik_elems if len(elem)>1]
        return [elem.strip() for elem in ik_elems if len(elem)]
    
# flatten lists of lists
def flatten_lol(t):
    return list(itertools.chain.from_iterable(t))
flatten_lol([[1,2],[3],[4,5,6]])

def print_elapsed(t0_local, task = 'current task'):
    print('Done with {}. Elapsed time: {:4f}'.format(task,time.time()-t0_local))

# define which data to load
qualifier = 'postjoining_final_no_dropped'
qualifier0 = 'postjoining_final_no_dropped'
pofs = 'n'

output_dir = '/Users/stefgarasto/Google Drive/Documents/results/NOS/nlp_analysis/'

lookup_dir = '/Users/stefgarasto/Google Drive/Documents/results/NOS/extracted/'

# save directory and files
nos_json_dir = '/Users/stefgarasto/Local-Data/json_files_NOS'
#'/Users/stefgarasto/Google Drive/Documents/Outputs/NOS_summary/json_files'
# NOTE: I might have to divide the NOS following the same
# directory structure I got them from (new NOS X / old NOS X)
nos_json_fname = '{}.json'

# Load the input directory structure
input_nos_dir = '/Users/stefgarasto/Google Drive/Documents/data/NOSdata'
nos_subdir = ['New NOS 1', 'New NOS 2']
nos_subfiles = [os.listdir(os.path.join(input_nos_dir,nos_subdir[0])), 
                os.listdir(os.path.join(input_nos_dir,nos_subdir[1]))]
for ix in range(1,23):
    nos_subdir.append('Old NOS {}'.format(ix))
    nos_subfiles.append(os.listdir(os.path.join(input_nos_dir,nos_subdir[-1])))

# make everything lower case
for ix in range(len(nos_subfiles)):
    nos_subfiles[ix] = [t.lower() for t in nos_subfiles[ix]]

#%%    
#Get the NOS data for approved apprenticeship standards from api
df_nos = pd.read_pickle(lookup_dir + 'all_nos_input_for_nlp_{}.zip'.format(qualifier0))
# load the cleaned and tokenised dataset
df_nos = df_nos.join(pd.read_pickle(lookup_dir + 'all_nos_input_for_nlp_{}_pruned_{}.zip'.format(
        qualifier,pofs)))
print('Done')

#%%
paths_dict = {}
# Find where each NOS is in the directory structure
for index,row in df_nos.iterrows():
    #print(row['Path'])
    found = False
    for ix in range(len(nos_subfiles)):
        # look by title
        if index in nos_subfiles[ix]:
            #print(index,ix)
            found = True
            paths_dict[index] = os.path.join(nos_json_dir, nos_subdir[ix],
                                    os.path.splitext(index)[0])
    if not found:
        # look by urn
        for ix in range(len(nos_subfiles)):
            if (row['URN']+'.pdf') in nos_subfiles[ix]:
                #print(index,row['URN'],ix)
                paths_dict[index] = os.path.join(nos_json_dir, nos_subdir[ix],
                                        row['URN']+'.pdf')
                found = True
    if not found:
        paths_dict[index] = os.path.join(nos_json_dir, row['URN']+'.pdf')


#%%
# manually remove "k"s and "p"s from the pruned columns
def remove_pk(x):
    if isinstance(x,float):
        return ['']
    if isinstance(x,list):
        #if len(x)>1:
        #    x = ' '.join(x).split()
        y=[]
        for item in x:
            y.append(' '.join([t for t in item.split() if t not in ['k','p']]))
        return y
    if isinstance(x,str):
        return [t for t in x if t not in ['k','p']]
#df_nos['pruned'] = df_nos['pruned'].map(remove_pk)

all_columns = df_nos.columns

for col in ['Overview','Performance_criteria','Knowledge_and_understanding',
            'Scope_range','Values','Behaviours','Skills','Glossary',
            'Links_to_other_NOS','External_Links']:
    df_nos[col] = df_nos[col].map(remove_pk)
    assert(isinstance(df_nos[col].iloc[0],list))

# for sections not in the json, do i want to include them? YES
# add sections that will always be empty
add_empty_columns = ["Scope_range_related_to_performance_criteria",
                     "Scope_range_related_to_knowledge_and_understanding"]
for col in add_empty_columns:
    df_nos[col]= ''
    df_nos[col] = df_nos[col].map(lambda x: [''])

# Need to turn some columns into a list of strings
#def to_list_of_string(x):
#    pass

def float_to_empty_list(x):
    if isinstance(x,float):
        return ['']
    else:
        return x

def float_to_empty_string(x):
    if isinstance(x,float):
        return ''
    else:
        return x
    
def date_to_str(x):
    try:
        x=x.strftime('%d %b %Y')
        if '1905' in x:
            x = ''
    except:
        x= ''
    return x

# transform specific columns to list of string
df_nos['Keywords'] = df_nos['Keywords'].map(get_keywords_list)
df_nos['Occupations']= df_nos['Occupations'].map(get_keywords_list)
# transform to '-delimited string
df_nos['Clean SOC Code'] = df_nos['Clean SOC Code'].map(lambda x: ';'.join(x) 
                    if isinstance(x,list) else '')

#Version_number <class 'numpy.float64'> !!!(change to string)
df_nos['Version_number'] = df_nos['Version_number'].map(lambda x: str(x))

#Indicative Review Date <class 'pandas._libs.tslibs.timestamps.Timestamp'>!!! (change to string)
df_nos['Indicative Review Date']= df_nos['Indicative Review Date'].map(date_to_str)

#Status <class 'str'>  !!!(replace null values with those from NOS Document Status)
nullrows = df_nos['Status'].isnull()
df_nos['Status'][nullrows] = df_nos['NOS Document Status'][nullrows]

df_nos['NOS Title']= df_nos['NOS Title'].map(lambda x: x.strip())

#%%
# fill in empty values appropriately
for col in ['Overview','Performance_criteria','Knowledge_and_understanding',
            'Scope_range','Values','Behaviours','Skills','Glossary',
            'Links_to_other_NOS','External_Links','Occupations','Keywords','All_suites']:
    df_nos[col] = df_nos[col].map(float_to_empty_list)
    
list_of_noise = ['\u2013','\uf0b7','\u2019','\u00e2','\u20ac','\u2122']
for col in ['Overview','Performance_criteria','Knowledge_and_understanding',
            'Scope_range','Values','Behaviours','Skills','Glossary',
            'Links_to_other_NOS','External_Links']:
    for token in list_of_noise:
        df_nos[col] = df_nos[col].map(lambda x: [t.replace(token,'') for t in x])

for col in ["Developed By",  "Version_number",  "Date_approved",
  "Indicative Review Date", "Validity", "Status", "Originating_organisation",
  "Original URN",'NOSCategory']:
    df_nos[col]= df_nos[col].map(float_to_empty_string)

# capitalize
for col in ['NOS Title','Validity','Status','Developed By','NOSCategory','Originating_organisation']:
    df_nos[col]= df_nos[col].map(lambda x: x.capitalize())

for col in ['All_suites','Keywords']:
    df_nos[col] = df_nos[col].map(lambda x: [t.capitalize() for t in x])


#%%
# turn each row into a json
# which columns do we want to include?
columns_of_interest = {'URN': 'URN', 'NOS Title':'Title', 
        'Overview': 'Overview','Performance_criteria': 'Performance_criteria',
       'Knowledge_and_understanding': 'Knowledge_and_understanding', 
       'Scope_range': 'Scope_range',
       "Scope_range_related_to_performance_criteria": 
               "Scope_range_related_to_performance_criteria",
       "Scope_range_related_to_knowledge_and_understanding":
               "Scope_range_related_to_knowledge_and_understanding",
       'Values': 'Values', 'Behaviours': 'Behaviours',
       'Skills': 'Skills', 'Glossary': 'Glossary',
       'Links_to_other_NOS': 'Links_to_other_NOS',
       'External_Links': 'External_Links',
       'Developed By': 'Developed_by',
       'Version_number':'Version_number',
       'Date_approved': 'Date_approved',
       'Indicative Review Date': 'Indicative_review_date',
       'Validity': 'Validity',
       'Status': 'Status',
       'Originating_organisation': 'Originating_organisation',
       'Original URN': 'Original_URN',
       'Relevant_occupations': 'Occupations',
       'All_suites': 'Suite',
       'Keywords': 'Keywords', 
       'Clean SOC Code': 'SOC_Code',
       'NOSCategory': 'NOSCategory'}

df_nos = df_nos[list(columns_of_interest.keys())]
df_nos = df_nos.rename(columns = columns_of_interest)

#%%
#def row_to_dict(row_local):
#  # DO STUFF
#  pass
#  return out_dict

# turn each row into a dict, then save as json
for index,df_row in df_nos.iterrows(): #CHECK!
  df_dict = df_row.to_dict()
  local_path = paths_dict[index]
  if local_path[-4:] == '.pdf':
      local_path = local_path[:-4]
  json_to_save =  local_path + '.json'
  with open(json_to_save, 'w') as fp:
    json.dump(df_dict, fp, indent=4, separators=(',', ': '))
  #df_row.to_json(json_to_save, indent=4, separators=(',', ': '))
  
  
  
  
#%%
  '''
# Small changes to the mismatches
basename = ''.join(['/Users/stefgarasto/Google Drive/Documents/Outputs/',
                    'NOS_summary/WP1_files/mismatches_pdf_activeis_{}.csv'])
basename_new = ''.join(['/Users/stefgarasto/Google Drive/Documents/Outputs/',
                    'NOS_summary/WP1_files/mismatches_pdf_activeis_new_{}.csv'])
A = ['Developing organisation','Status','Title','Suite']
for ix,t in enumerate(['Developed_by','Status','Title','Clean_suite']):
    df_mismatch = pd.read_csv(basename.format(t))
    col = '{} extracted from pdf'.format(A[ix])
    print(len(df_mismatch))
    df_mismatch = df_mismatch[df_mismatch[col].map(lambda x: x!='empty')]
    df_mismatch = df_mismatch[~df_mismatch[col].isnull()]
    df_mismatch['URN'] = df_nos['URN'].loc[[t for t in df_mismatch['NOS file name']]].values
    print(len(df_mismatch))
    df_mismatch.to_csv(basename_new.format(t))
'''