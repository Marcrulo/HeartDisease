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
import scipy
from scipy.stats import kstest, norm, zscore
import statsmodels.api as sm
import pylab

# %% Import data

# Read
#my_data = np.genfromtxt('heart.csv', names=header, delimiter=',')
df=pd.read_csv('heart.csv')

# Split
data=df.loc[:,'age':'thal']
target=df.iloc[:,-1]

# %% Summary statistics

# MEASURES OF UNCERTAINTY
print(df.mean()) # mean
print(df.median()) # median
print(df.quantile(0.25)) # 25% quantile
print(df.quantile(0.50)) # 50% quantile
print(df.quantile(0.75)) # 75% quantile

# MEASURES OF SPREAD
print(df.var()) # variance
print(df.std()) # std. dev.
print(df.std()/df.mean()) # <--
print(df.quantile(0.75)-df.quantile(0.25)) # IQR

# MEASURES OF RELATION
print(df.cov()) # cov
print(df.corr()) # corr
correlationMatrix = df.corr() 


# 'mixed'
print(df.describe(include='all'))






### Suggestion: Find a way to find outliers and comment on them
### Suggestion: Find out if the attributes are normally distributed


# %% Data distribution - Visualization

N = df.shape[0]
M = df.shape[1]

X = df.to_numpy()

attributeNames = list(data.columns)

plt.figure(figsize=(6,4))
u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
for i in range(M-1):
    plt.subplot(u,v,i+1)
    plt.hist(X[:,i], color=(0.2, 0.8-(i/3)*0.2, 0.4))
    plt.xlabel(attributeNames[i])
    plt.ylim(0,N/2)
    plt.xticks([])
    plt.yticks([])
    
plt.show()


# %% Data distribution - Mathematics
### Source: https://towardsdatascience.com/6-ways-to-test-for-a-normal-distribution-which-one-to-use-9dcf47d8fa93


cont_attrs = ["age", "trestbps", "chol", "thalach", "oldpeak"]

for i in range(len(cont_attrs)):
    my_data = zscore(df[cont_attrs[i]], ddof=1)
    sm.qqplot(my_data, line='45')
    ks_statistic, p_value = [round(i, 3) for i in kstest(my_data, 'norm')]
    print("{} has a k statistic of {} and a p value of {}.".format(cont_attrs[i], ks_statistic, p_value))
    pylab.show()


# %% Box plots for anomaly detection

### Just an overview of all attributes
### Standardize the continous attributes and remove the nominal ones to get
### a feasible comparision of alle attributes

plt.boxplot(zscore(X, ddof=1))
plt.xticks(range(1,14),attributeNames,rotation=90)

plt.ylabel('')
plt.title('Heart disease dataset - chaotic boxplot')
plt.show()


### Individual boxplot of the five continous attributes
### age, trestbps, chol, thalach, oldpeak

X_tilde = np.array([X[:, 0], X[:, 3], X[:, 4], X[:, 7], X[:, 9]])
attributeNames_tilde = [attributeNames[0], attributeNames[3], attributeNames[4], 
                        attributeNames[7], attributeNames[9]]



for i in range(0, X_tilde.shape[0]):
    plt.boxplot(X_tilde[i].T)
    plt.xlabel(attributeNames_tilde[i])
    plt.ylabel('')
    plt.title('Heart disease dataset - boxplot')
    plt.show()



# %% Data matrix

X_standarized = zscore(X, ddof=1)

plt.figure(figsize=(12,6))
plt.imshow(X_standarized, interpolation='none', aspect=(4./N), cmap=plt.cm.gray);
plt.xticks(range(13), attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Data objects')
plt.title('Heart disease data matrix')
plt.colorbar()

plt.show()



# %% PCA
pca = PCA(n_components=5)
principalComponents = pca.fit_transform(df)

principalDf = pd.DataFrame(data = principalComponents)

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

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

# %% Direction of PCA components
print(pca.components_)

N,M=df.shape
V=pca.components_.T

pcs = [0,1,2,4]
legendStrs = ['PC'+str(e+1) for e in pcs]
bw = .2
r = np.arange(1,M+1)
attributeNames = df.columns.values.tolist()

for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames,rotation=90)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('NanoNose: PCA Component Coefficients')
plt.show()



# %% PLOT 2 FIRST PRINCIPCAL COMPONENTS
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['principal component 1', 'principal component 2']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


