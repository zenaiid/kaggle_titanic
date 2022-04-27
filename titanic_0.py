# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 23:44:09 2022

@author: Tomasz
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./data/train.csv', delimiter = ',')
# print(df.head())
print(df.describe())
print(f'shape: {df.shape}')

# correlation matrix as heatmap
corr_df = df.corr()

plt.figure(figsize=(13, 6))
sns.heatmap(corr_df, vmax=1, annot=True, linewidths=.5)
plt.xticks(rotation=30, horizontalalignment='right')
plt.show()

print(df['Pclass'].value_counts())
print(df['Embarked'].value_counts())

print(pd.get_dummies(df['Embarked']))

print(df.shape)
# 'Pclass sollte ich auch one hot encoden
df = pd.concat([df, pd.get_dummies(df['Embarked'])], axis=1)
df = pd.concat([df, pd.get_dummies(df['Pclass'])], axis=1)
#
df = df.drop(columns=['Embarked', 'Pclass'])
print(df.shape)

# correlation matrix as heatmap
corr_df = df.corr()

plt.figure(figsize=(13, 6))
sns.heatmap(corr_df, vmax=1, annot=True, linewidths=.5)
plt.xticks(rotation=30, horizontalalignment='right')
plt.show()

# exploratory data analysis

# prepare date
# feature engineering
# - kick out name?
# lock at correlation between features
# what about cabin column?
# normalize data

# retrieve Mrs / Miss information from name column?
# what does second name im brackets mean?

# one hot encode e.g. embark
# possible new features

# SVM 
# logistic regression
# knn
# random forest
# neural network

# cross validation