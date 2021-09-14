#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 15:49:55 2021

@author: marc8165
"""

# %% Import packages/libraries
import numpy as np
import pandas as pd



# %% Import data

# Read
df = pd.read_csv('heart.csv')
print(df.head())

# Split
data=df.loc[:,'age':'thal']
target=df.iloc[:,-1]


# %% Summary statistics

# MEASURES OF UNCERTAINTY
print(df.mean()) # mean
print(df.median()) # median
print('quantiles') # quantiles

# MEASURES OF SPREAD
print(df.var()) # variance
print(df.std()) # std. dev.
print('coefficient of variation') # <--
print('inter quartile range') # IQR

# MEASURES OF RELATION
print('covariance') # cov
print('correlation') # corr


# 'mixed'
print(df.describe(include='all'))



# %% PCA or something...






# %%








