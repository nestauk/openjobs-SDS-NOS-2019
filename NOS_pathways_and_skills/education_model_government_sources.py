#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:25:03 2019

@author: stefgarasto


'''
This script is to apply a model that links SOC codes to educational requirements
based on official goverment classification (derived from immigration rules
Appendix J: codes of practice for skilled work: 
https://www.gov.uk/guidance/immigration-rules/immigration-rules-appendix-j-
codes-of-practice-for-skilled-work)
'''
There are 8 SOC codes that can't be classified using this model:  
'3133', '3442', '8140', '1170', '9'.
Of these, 4 have been sorted in bg_solve_specific_soc
Only 3442 is not matched and has examples of educational requirements.
This SOC code correspond to sports coaches, instructors and officials.
There are 6290 rows with this SOC code that also have educational requirements.
It makes for a total of 719 job titles.
4999 --> pregraduate
1222 --> graduate
69 --> postgraduate

Also, the three categories have overlapping job titles, so I can't categorize
by job title.

The quickest compromise would be to assign the missing soc code to "Pregraduate",
but I would lose a lot of signal this way.
I'll use this so far and then we'll see

Note: the other main problem is that the occupations that are assigned to level
NQF 6 are actually for level NQF 6 or above (basically RQF 6 or 7 since the PhD
ones are well defined). However, all the SOC codes matched to RQF 6 that are 
relevant for any of the super-suites are matched to Graduate in the majority
of cases. So I guess that the main problem with this model is the LACK OF
GRANULARITY: SOC CODES ARE WAY COARSER THAT JOB TITLES + SKILLS!

This script assumes that "bg_load_and_prep_data" has been run already
"""

#%%
from utils_bg import *
import copy
import re
from bg_solve_specific_soc_codes import matches_oobsoc_to_soc

'''
#%% reduce the dataset further to keep only SOC codes relevant to the 
# super-suites
t0 = time.time()
SUPER = 'all'
if SUPER == 'all':
    bgdatasmall = bgdatared[bgdatared['SOC'].astype(int).map(
                lambda x: str(x) in total_socs4)]
elif SUPER == 'none':
    bgdatasmall = bgdatared
else:
    bgdatasmall = bgdatared[bgdatared['SOC'].astype(int).map(
                lambda x: str(x) in soc4dist[SUPER])]
print_elapsed(t0, 'selecting SOCs relevant to super-suites')
'''

#%%
''' Load the table with the official matches and build the dictionary''' 
fname = '/Users/stefgarasto/Google Drive/Documents/data/Burning-glass/soc_to_nqf.xlsx'

df_table = pd.read_excel(fname, sheet_name = 'Table')
# remove rows that are not necessary
df_table= df_table[~df_table['SOC and name'].isnull()]
# extract SOC code
df_table['SOC']= df_table['SOC and name'].map(lambda x: re.findall('\d{4}',x)[0])

official_soc_matches = {}
for ix in df_table.index:
    if df_table['Skill level'].loc[ix] == 'PhD':
        category = 'Postgraduate'
    elif df_table['Skill level'].loc[ix] == 'RQF 6':
        category = 'Graduate'
    else:
        category = 'Pregraduate'
    official_soc_matches[df_table['SOC'].loc[ix]] = category

#%% 
# Check which SOCs are missing
A_table = list(official_soc_matches.keys()) #df_table['SOC'].values)
# get all the socs
K = list(socnames_dict.keys())
K = ['{:.0f}'.format(t) for t in K if len('{:.0f}'.format(t)) == 4]
# get the socs I need
socs_i_need = ['{:.0f}'.format(t) for t in list(bgdatasmall['SOC'].value_counts().index)]

#%% all the SOCs I need are in total_socs4
missing_socs_from_nos = [t for t in total_socs4 if t not in A_table]
missing_socs_from_all = [t for t in K if t not in A_table]
missing_socs_from_bg = [t for t in socs_i_need if t not in A_table]

#%% check distribution of missing socs
official_soc_matches_final = copy.deepcopy(official_soc_matches)
for isoc in missing_socs_from_nos:
    if isoc in matches_oobsoc_to_soc:
        print('all good with {}'.format(isoc))
        continue
    print('SOC code {} needs to be sorted'.format(isoc))
    tmp = bgdatasmall[bgdatasmall['SOC']==float(isoc)]['Edu'].value_counts()
    #print(tmp)
    # Quick fix: simply match it to the most common requirement
    official_soc_matches_final[isoc] = tmp.idxmax()

#%%
''' Just to be totally on the safe side I'll classify all missing SOCs, even
if they are not relevant to the super-suites. It doesn't take long to classify
them using MAP anyway (max across the category distribution for that SOC code)
'''
for isoc in missing_socs_from_all:
    print('SOC code {} needs to be sorted'.format(isoc))
    tmp = bgdatasmall[bgdatasmall['SOC']==float(isoc)]['Edu'].value_counts()
    if len(tmp):
        # if it can be found among the super-suites, great
        official_soc_matches_final[isoc] = tmp.idxmax()
    else:
        # if not, classify from the larger BG dataset
        tmp = bgdatared[bgdatared['SOC']==float(isoc)]['Edu'].value_counts()
        official_soc_matches_final[isoc] = tmp.idxmax()
    print(isoc, tmp.idxmax())
    
with open(os.path.join(saveoutput,'government_based_model_Edu.pickle'), 'wb') as f:
    pickle.dump(official_soc_matches_final, f)    
    

