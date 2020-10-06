#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 12:17:33 2020

@author: stefgarasto
"""

from IPython.display import display, HTML
import itertools
import pandas as pd
from time import time as tt

#%% Timing functions
def print_elapsed(t0,task):
    """ Minimal function that prints how much time was spent on a task
    since a given starting point.

    Args:
    t0: task starting timer
    task: task description
    """
    print(f'Time spent on {task} took {(tt()-t0)/60:.2f} minutes.')

class TaskTimer:
    """
    This class keeps track of the time spent on a task.

    """
    def __init__(self):
        pass

    def start_task(self, task = ''):
        """ When called, it initialise a new task, by storing the time at which
        it started and the description.
        Args:
        task = task description.
        """
        self.t0 = tt()
        self.task = task
        print(f"Starting to work on: '{self.task}'")

    def end_task(self):
        """
        Shows how much time passed since the beginning of the task
        """
        print_elapsed(self.t0, self.task)

# define official Nesta colours
nesta_colours= [[1, 184/255, 25/255],[1,0,65/255],[0,0,0],
    [1, 90/255,0],[155/255,0,195/255],[165/255, 148/255, 130/255],
[160/255,145/255,40/255],[196/255,176/255,0],
    [246/255,126/255,0],[200/255,40/255,146/255],[60/255,18/255,82/255]]

# Add primary colours if longer colours list  is needed
aug_nesta_colours = nesta_colours + [[1., 0., 0.], [0., 0., 1.], [0.,1.,0.], [.7, .7, .7]]

# Nesta colour combos. The lists are for:
# primaries, secondaries, bright combination, warm combination
# cool combination, neutral with accent colour combination,
# deep and accent colour combination.
# The numbers in each list indicate the position of the colours within the nesta_colours list
nesta_colours_combos = [[0,1,2,3,4,5],[0,6,7],[1,3,8],
                [4,9,10],[8,5],[1,11]]

# Helper functions
def flatten_lol(input_list):
    """ flatten lists of lists

    Args:
    input_list: list
    """
    return list(itertools.chain.from_iterable(input_list))

def flattencolumns(df, cols = ['alt_labels']):
    """ This function flattens data frame columns that are lists.
    That is, it spreads a column of lists across multiple columns.

    Args:
    df: dataframe
    cols: names of columns to flatten
    """
    df1 = pd.concat([pd.DataFrame(df[x].values.tolist()).add_prefix(x)
                    for x in cols], axis=1)
    return pd.concat([df1, df.drop(cols, axis=1)], axis=1)

def printdf(df):
    """ print out a dataframe """
    display(HTML(df.to_html()))

#%% Labour market info that is used on many occasions
# SIC codes
sic_letter_to_text = {'A': 'Agriculture, forestry and fishing',
                      'B': 'Mining and quarrying',
                      'C': 'Manufacturing',
                      'D': 'Electricity, gas, steam and air conditioning supply',
                      'E': 'Water supply; sewerage, waste management and remediation activities',
                      'F': 'Construction',
                      'G': 'Wholesale and retail trade; repair of motor vehicles and motorcycles',
                      'H': 'Transportation and storage',
                      'I': 'Accommodation and food service activities',
                      'J': 'Information and communication',
                      'K': 'Financial and insurance activities',
                      'L': 'Real estate activities',
                      'M': 'Professional, scientific and technical activities',
                      'N': 'Administrative and support service activities',
                      'O': 'Public administration and defence; compulsory social security',
                      'P': 'Education',
                      'Q': 'Human health and social work activities',
                      'R': 'Arts, entertainment and recreation',
                      'S': 'Other service activities',
                      'T': 'Activities of households as employers; undifferentiated goods-and services-producing activities of households for own use',
                      'U': 'Activities of extraterritorial organisations and bodies'}

# SOC codes descriptions
# CHANGE THE FILE PATH TO WHEREVER IT'S STORED
socnames_file = '/Users/stefgarasto/Google Drive/Documents/data/ONS/' \
    + 'soc2010indexversion705june2018.xls'
# Read the file describing SOC codes
socnames = pd.read_excel(socnames_file, sheet_name = 'SOC2010 Structure')
soccolnames = {1: 'Major Group', 2:'Sub-Major Group', 3: 'Minor Group',
               4: 'Unit   Group'}
# create dictionary from SOC codes at all different levels to SOC labels
socnames_dict = {}
for digit in range(1,5):
    all_socs = socnames[soccolnames[digit]].value_counts().index
    all_socs = [int(t) for t in all_socs]
    for isoc in all_socs:
        k = socnames[soccolnames[digit]]==isoc
        socnames_dict[isoc] = socnames['Group Title'][k].values[0].capitalize()
        
