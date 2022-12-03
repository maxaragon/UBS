#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.metrics import roc_auc_score, average_precision_score,roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

#def eer(fpr, tpr):
#    return brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

def eer(y_true,y_pred):
    fpr, tpr, threshold = roc_curve(y_true, y_pred, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    return fpr[np.nanargmin(np.absolute((fnr - fpr)))]