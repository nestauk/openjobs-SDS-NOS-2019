#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 08:24:23 2019

@author: jdjumalieva

PREVIOUS VERSION of scrape_nos.py: please see "scrape_nos.py" for information.
Better not to use this one, it is kept mostly for reference.

"""

import os
from collections import defaultdict
import textract


input_dir = '/Users/jdjumalieva/Documents'

standard_info = defaultdict(dict)

#Prep the pdf##################################################################
#Convert pdf to bytes
pdftext = textract.process(os.path.join(input_dir, 'sample_standard.pdf'))

#Convert bytes to string
rawtext = pdftext.decode('utf-8')

#Get standard reference id
standard_ref = rawtext.split('\n')[0]

#Split content by standard reference id (this should get us standard pages)
tables = rawtext.split(standard_ref)

#Split by new lines and remove empty elements
nested_tables = [elem.split('\n') for elem in tables if len(elem)]
nested_tables2 = [[subelem for subelem in elem if len(subelem)] \
                   for elem in nested_tables]

#Extract fields from standard content##########################################
standard_name = nested_tables2[0][0]

to_remove = [standard_name, 'Overview'] #field names which we'll ignore
standard_overview = [elem for elem in nested_tables2[0] if elem not in \
                     to_remove]

standard_info[standard_ref]['standard_name'] = standard_name
standard_info[standard_ref]['standard_overview'] = standard_overview

to_remove2 = [standard_name, 'Performance criteria', 'You must be able to:']
standard_perf = [elem for elem in nested_tables2[2] if elem not in to_remove2]

knowledge_ix = nested_tables2[5].index(standard_name) 
standard_knowledge = nested_tables2[5][:knowledge_ix]

standard_info[standard_ref]['standard_perf'] = standard_perf
standard_info[standard_ref]['standard_know'] = standard_knowledge

meta_data = nested_tables2[6] + nested_tables2[7] 

standard_info[standard_ref]['developed_by'] = \
meta_data[meta_data.index('Developed by')+1]

standard_info[standard_ref]['verion_num'] = \
meta_data[meta_data.index('Version Number')+1]

standard_info[standard_ref]['date_approved'] = \
meta_data[meta_data.index('Date Approved')+1]

standard_info[standard_ref]['indic_review_date'] = \
meta_data[meta_data.index('Indicative Review')+1]

standard_info[standard_ref]['validity'] = \
meta_data[meta_data.index('Validity')+1]

standard_info[standard_ref]['status'] = \
meta_data[meta_data.index('Status')+1]

standard_info[standard_ref]['orig_org'] = \
meta_data[meta_data.index('Originating')+1]

#Need to check if this is always the same as standard reference id
standard_info[standard_ref]['orig_urn'] = \
standard_ref

standard_info[standard_ref]['date_approved'] = \
meta_data[meta_data.index('Date Approved')+1]

occup_list = meta_data[meta_data.index('Relevant'): meta_data.index('Suite')]
to_remove3 = ['Relevant', 'Occupations']
occupations = ' '.join([elem for elem in occup_list if elem not in to_remove3])

standard_info[standard_ref]['relev_occup'] = occupations

standard_info[standard_ref]['suite'] = \
meta_data[meta_data.index('Suite')+1]

standard_info[standard_ref]['keywords'] = \
meta_data[meta_data.index('Keywords')+1]

#file_path = '/Users/jdjumalieva/Downloads/'
#standard_tables = pd.read_html(os.path.join(file_path, 'sample_standard.html'))
#
#
#standard_info = defaultdict(dict)
#table1 = standard_tables[0]
#
#standard_ref = table1.loc[0,0]
#standard_name = table1.loc[1,0]
#
#table1 = table1[:-1]
#standard_overview = ' '.join(table1[1].dropna().values)
#
#standard_info[standard_ref]['standard_name'] = standard_name
#standard_info[standard_ref]['standard_overview'] = standard_overview
#
#table2 = standard_tables[1]
#performance = table2[1][4:-1]
#performance_list = ' '.join(performance.values)
#
#result = re.split('([0-9]+)', performance_list)
#result = [elem for elem in result if len(elem)]
#
#numbers = result[0::2]
#items = result[1::2]
#
#combined = list(zip(numbers, items))
#
#combined_tidied = [''.join(elem) for elem in combined]
#
#
#table4 = standard_tables[3]
#for ix, row in table4[4:-1].iterrows():
#    print('key', row[0], ':', 'value', row[1])