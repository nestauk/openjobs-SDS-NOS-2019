#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:53:54 2019

@author: stefgarasto
"""

import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier

#%%
'''
# To use random forest with inbuilt resampling (undersampling
# though)
brf = BalancedRandomForestClassifier(n_estimators=100)
for train, test in cv.split(FEATURES, TARGETS):
    brf.fit(FEATURES[train], TARGETS[train])
    y_pred_brf = brf.predict(FEATURES[test])
    probas_ = brf.predict_proba(FEATURES[test])
'''

# to compute prediction intervals for random forest classifier
def pred_ints(pred_model, X, percentile=95):
    err_down = []
    err_up = []
    for x in range(len(X)):
        preds = []
        for pred in pred_model.estimators_:
            preds.append(pred.predict(X[x])[0])
        err_down.append(np.percentile(preds, (100 - percentile) / 2. ))
        err_up.append(np.percentile(preds, 100 - (100 - percentile) / 2.))
    return err_down, err_up


'''
Other useful things.
1. To use an ensemble classifier with inbuilt class balancing. It always uses
undersampling though

from imblearn.ensemble import BalancedBaggingClassifier
bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                 sampling_strategy='auto',
                                 replacement=False,
                                 random_state=0)

To train a balances RandomForestClassifier:
from imblearn.ensemble import BalancedRandomForestClassifier
brf = BalancedRandomForestClassifier(n_estimators=100, random_state=0)
brf.fit(X_train, y_train) 
BalancedRandomForestClassifier(...)
y_pred = brf.predict(X_test)


'''

