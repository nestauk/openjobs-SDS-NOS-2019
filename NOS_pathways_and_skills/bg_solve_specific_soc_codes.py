#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:59:27 2019

@author: stefgarasto

The NOS database has some SOC codes that don't appear in the official 
classification.

Match them here as special cases

5134 was a mistake - should have been 5314
3524 also - should have been 3534
8148 should have been 8149

The remaining extra SOC codes are 3133, 8140, 1170, 9.
Notes for each one of them.

1. 3133 does not exist as SOC code. Technically the closest matches are 3131 and
3132. However, neither exists in the BG dataset. There is only one NOS with this
SOC code and the occupation is "Engineering Technicians". The SOC code for this
occupation is 3113, so I'm choosing to assume this is the right SOC code (which
does exist in the BG dataset).

2. In the NOS database, there is again just one NOS with 8140. The occupation is
construction operatives, so I think it should just be matched to 8149 (which exists
in the BG dataset)

3. For 1170, there are 103 NOS, so probably not a mistake. However, the BG dataset
with educational requirements only has information about SOC code 1173. Thus,
I'm matching 1170 to 1173.

4. One NOS has SOC code 9, but I think it's wrong because it's about hospitality
managers. The SOC code that would fit best is 1221, so for now I'm matching them
(and BG has 1221)
"""

#%%
from utils_bg import *

#%%
# get all the socs
K = list(socnames_dict.keys())
K = ['{:.0f}'.format(t) for t in K if len('{:.0f}'.format(t)) == 4]

#%%
oob_socs = [t for t in total_socs4 if t not in K]

#%%
matches_oobsoc_to_soc = {'3133': '3113', '8140': '8149','1170':'1173','9': '1221'}
