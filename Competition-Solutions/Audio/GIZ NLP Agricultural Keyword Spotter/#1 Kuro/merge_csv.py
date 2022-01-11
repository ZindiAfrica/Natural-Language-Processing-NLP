# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 20:37:03 2020

@author: Shiro
"""


import pandas as pd
import copy
import numpy as np



## put inside listes the csv file representing the prediction of a model
listes = ["model_1-5folds-CV-seed42-bs16-mixup.csv", "model_4-5folds-CV-seed42-bs16-mixup.csv",
          "model_6-5folds-CV-seed42-bs16-mixup.csv", "model_7-5folds-CV-seed42-bs16-mixup.csv"]
df = [pd.read_csv(l) for l in listes]
cols = df[0].columns[1:]
arr = np.stack([d[cols].values for d in df]).mean(0)

final = copy.deepcopy(df[0])
final[cols] = arr

# csv filename output 
final.to_csv("submission_ensemblingv5.csv", index=False)