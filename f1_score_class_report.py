#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 14:48:20 2019

@author: jdjumalieva
"""

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.metrics import recall_score, classification_report, confusion_matrix

predictions = clf3.predict(balanced_sample_matrix.toarray())

f1_score(list(balanced_sample['Target']), predictions, average='binary')

X = balanced_sample_matrix.toarray()
y = list(balanced_sample['Target'])
sss = StratifiedShuffleSplit(y, n_iter=1, test_size=0.5, random_state=0)

y2 = np.asarray(y)

for train_index, test_index in sss:
    X_train, X_test = X[train_index, :], X[test_index, :]
    y_train, y_test = y2[train_index], y2[test_index]
    clf3.fit(X_train, y_train)
    y_pred = clf3.predict(X_test)
    print(f1_score(y_test, y_pred, average="binary"))
    print(precision_score(y_test, y_pred, average="binary"))
    print(recall_score(y_test, y_pred, average="binary"))
    
    
print(classification_report(y_test, y_pred))
accuracy_score(y_test, y_pred)
