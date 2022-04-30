# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 17:23:38 2022

@author: Tomasz
"""
import re
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

def split_at_number(string, mode):
  items = [string]
  match = re.match(r"([a-z]+)([0-9]+)", string, re.I)
  if match:
      items = match.groups()
  if mode == 'deck':
    return items[0].split(' ')[0]
  elif mode == 'loc':
     if len(items) > 1:
        return int(items[1])
     else:
        return -100
    
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

def parse_surname(name):
  surname = []
  if "Mr." in name or "Master." in name or "Miss." in name:
    surname.append(name.split(',')[0])
  elif "Mrs." in name:
    surname.append(name.split(',')[0])
    surname.append(name.split(" ")[-1].strip(")"))
  return surname

def one_hot(colNames, df):
    for col in colNames:
        dummies = pd.get_dummies(df[col])
        df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=colNames)
    return df

def label_encode(values_dict, df, colName):
    for _, (key, value) in enumerate(values_dict.items()):
        df.loc[df[colName]==key, colName] = value
    return df

def feature_engineering(df):
    # returns, cleaned and enriched dataframe
    
    df["Cabin"] = df["Cabin"].fillna("?")
    df["deck"] = df['Cabin'].apply(split_at_number, args=('deck',))
    df["cNo"] = df['Cabin'].apply(split_at_number, args=('loc',))
    
    df["Tinfo"] = df['Ticket'].apply(get_ticket_info)
    df["Tinfo"] = df['Tinfo'].apply(clean_string)
    df["Tno"] = df['Ticket'].apply(get_ticket_no)
    
    # label encode 'Embarked' in the order of stops
    df = label_encode({'S': 1, 'C': 2, 'Q': 3}, df, 'Embarked')
    deck_encode = {'?': -100, 'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7,
                   'T':8}
    df = label_encode(deck_encode, df, 'deck')
    
    # deal with nan's:
    #TODO: evaluate other ways to deal with Age nan's! for example use mean / mode specific to male / female
    # values = {'Age': int(df['Age'].mean()), 'Embarked': df['Embarked'].mode()}
    values = {'Age': int(df['Age'].mean()), 'Embarked': 1}
    df = df.fillna(value=values)

    # change type
    types = {'Embarked': 'int64', 'deck': 'int64'}
    df = df.astype(types)
    
    # one hot encode
    df = one_hot(['Sex'], df)
    
    # drop
    df = df.drop(columns = ['Name', 'Cabin', 'Tinfo', 'Ticket', 'PassengerId'])
    return df

    #TODO: maybe also drop columns that have very small value count after one hot encoding????
                 
     #TODO: find spouses, sibllings, parents and children - create feature which is
     # 1 when your the oldest and zero if you are the youngest among them,
     # linear inbetwee

     #TODO: furhter clean up Tinfo column, maybe research what abbreviations could mean

     #TODO: maybe add boolean column 'spouse aboard'
     #TODO: maybe enrich Sex column, with female Mrs. / Miss etc.     

    #TODO: add parameters for feature engineering function           
                 
if __name__ == '__main__':
     df = pd.read_csv('./data/train.csv', delimiter = ',')
     df = feature_engineering(df)
     
     # correlation matrix as heatmap
     corr_df = df.corr()

     plt.figure(figsize=(13, 6))
     sns.heatmap(corr_df, vmax=1, annot=True, linewidths=.5)
     plt.xticks(rotation=30, horizontalalignment='right')
     plt.show()
     # print(df[['?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']].sum())
     # to not blow up, dimensions it could be wise to label encode the deck info
     # for example from top (A=1) to bottom (T=8) but then what to do with the 
     # large number of unknown cabins???
     # print(pd.get_dummies(df['Sex']))