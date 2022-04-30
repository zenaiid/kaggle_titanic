# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 23:44:09 2022

@author: Tomasz
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

df = pd.read_csv('./data/train.csv', delimiter = ',')
# print(df.head())
print(df.describe())
print(f'shape: {df.shape}')

#ToDo: use implemented k-fold cross validation
# fillna values, of embarked. Leave Priceclass as it is (?)
# label encoding vs one hot encoding of embarked (after all, there is a natural order in embarked)
# one hot encode male / female - differentiate between male Mrs. / Miss ? 
# implement pytorch dataset - 

# ticket seems to hold some information, extract all non numeric categories of ticket column, 
# how many are there, can they be categorized? Is ticket-nr. relevant at all? could be



# first easy model: logistic regression

# look more closely at data, for example look at fares, where price category and embarked are the same
# barplots of all categorical data

# correlation matrix as heatmap
corr_df = df.corr()

plt.figure(figsize=(13, 6))
sns.heatmap(corr_df, vmax=1, annot=True, linewidths=.5)
plt.xticks(rotation=30, horizontalalignment='right')
plt.show()

print(df['Pclass'].value_counts())
print(df['Embarked'].value_counts())

print(pd.get_dummies(df['Embarked']))

def split_at_number(string, mode):
  items = [string]
  match = re.match(r"([a-z]+)([0-9]+)", string, re.I)
  if match:
      items = match.groups()
  if mode == 'deck':
    return items[0]
  elif mode == 'loc':
     if len(items) > 1:
        return int(items[1])
     else:
        return -100

df["Cabin"] = df["Cabin"].fillna("?")
df["deck"] = df['Cabin'].apply(split_at_number, args=('deck',))
df["cNo"] = df['Cabin'].apply(split_at_number, args=('loc',))
df = df.drop(columns='Cabin')

# correlation matrix as heatmap
corr_df = df.corr()

plt.figure(figsize=(13, 6))
sns.heatmap(corr_df, vmax=1, annot=True, linewidths=.5)
plt.xticks(rotation=30, horizontalalignment='right')
plt.show()

def get_ticket_info(string):
  info = string.split(" ")
  if len(info) > 1:
    return info[0]
  else:
    return "?"

def get_ticket_no(string):
  info = string.split(" ")
  if len(info) > 1 and info[-1].isnumeric():
    return int(info[-1])
  elif info[0].isnumeric():
    return int(info[0])
  else:
      return 0

def clean_string(string):
  string = string.upper()
  if 'SOTON' in string:
    string = string.replace("SOTON", "STON")
  return string.replace('.', '')

 
# for idx in range(len(df)):
#   print(f"idx: {idx}, {get_ticket_no(df.loc[idx].Ticket)}")

df["Tinfo"] = df['Ticket'].apply(get_ticket_info)
df["Tinfo"] = df['Tinfo'].apply(clean_string)
df["Tno"] = df['Ticket'].apply(get_ticket_no)
df = df.drop(columns='Ticket')

def parse_surname(name):
  surname = []
  if "Mr." in name or "Master." in name or "Miss." in name:
    surname.append(name.split(',')[0])
  elif "Mrs." in name:
    surname.append(name.split(',')[0])
    surname.append(name.split(" ")[-1].strip(")"))
  return surname

for id in range(1, 5):
  print(df.loc[id-1].Name)
  print(f"id: {id} name: {parse_surname(df.loc[id-1].Name)}")
    
#TODO: find spouses, sibllings, parents and children - create feature which is
# 1 when your the oldest and zero if you are the youngest among them,
# linear inbetween
print(df[df['SibSp']+df['Parch']==0].count()/len(df))

#TODO: furhter clean up Tinfo column, maybe research what abbreviations could mean

#TODO: maybe add boolean column 'spouse aboard'
#TODO: maybe enrich Sex column, with female Mrs. / Miss etc.

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