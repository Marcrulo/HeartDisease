#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 15:49:55 2021

@author: marc8165
"""

# %% Import packages/libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

# WHAT NEEDS TO BE DESCRIBED
# the amount of variation explained as a function of the number of PCA components included
# principal directions of the considered PCA components
# the data projected onto the considered principal components



# %% PCA
pca = PCA(n_components=11)
principalComponents = pca.fit_transform(data)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

# %% PLOT 2 FIRST PRINCIPCAL COMPONENTS
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0,1]
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# %% # EXPLAINED VARIANCE
exp_var_pca = pca.explained_variance_ratio_
cum_sum_eigenvalues = np.cumsum(exp_var_pca)

# Individual explained variance
plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')

# Cumulative explained variance
plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()