#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 14:19:29 2019

@author: stefgarasto
"""

#%%
def check_dentistry(x):
    if isinstance(x,str):
        return ('orthodontics' in x) | ('dentistry' in x)
    else:
        return False

def check_edu20_titles(x):
    if isinstance(x,str):
        return x in ['orthodontist', 'dentist', 'clinical lecturer',
                     'clinical lecturer restorative dentistry',
                     'appointment service', 'specialty training']
    else:
        return False

def check_endodontics(x):
    if isinstance(x,str):
        return 'endodontics' in x
    else:
        return False
    
#%%
A = bgdatared[bgdatared['MinEdu']==20.0]

#%%
''' Check the distribution for dentist with dentistry. 
Can I just assign them to  MinEdu = 13? '''
D = bgdatared[(bgdatared['title_processed']=='dentist') & (bgdatared['converted_skills']=="['dentistry']")]

D['MinEdu'].value_counts()

#%%