#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 16:37:31 2019

@author: stefgarasto
"""

qualifier = 'postjoining_final_no_dropped'
qualifier0 = 'postjoining_final_no_dropped'
pofs = 'nv'

WHICH_GLOVE = 'glove.6B.100d' #'glove.6B.100d' #'glove.840B.300d', 
#glove.twitter.27B.100d

glove_dir = '/Users/stefgarasto/Local-Data/wordvecs/'

paramsn = {}
paramsn['ngrams'] = 'uni'
paramsn['pofs'] = 'nv'
paramsn['tfidf_min'] = 3
paramsn['tfidf_max'] = 0.5

paramsn['bywhich'] = 'docs' #'docs' #'suites'
paramsn['mode'] = 'tfidf' #'tfidf' #'meantfidf' #'combinedtfidf' #'meantfidf'