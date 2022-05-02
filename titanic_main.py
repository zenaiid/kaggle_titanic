# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 17:30:56 2022

@author: Tomasz
"""
import pandas as pd
from feature_engineering import feature_engineering
#import numpy as np
from toch.utils.data import Dataset, DataLoader
import torch
from torch import nn, optim

df = pd.read_csv('./data/train.csv', delimiter = ',')
df = feature_engineering(df)
# print(df.dtypes)

y = torch.tensor(df['Survived'].values)
X = torch.tensor(df.loc[:, df.columns != 'Survived'].values)

#TODO: put model definitions in function in seperate file 
# define model
class LogisticRegression(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        out = self.linear(x)
        return out

#TODO: Datasets und Dataloader:
# 1. define test and train datasets where csv's are loaded (outside k-fold loop)
# 2. define dataloader (outside k-loop)
# 3. define subsamplers for k-fold cross validation (inside k-loop because dependent on subsamplng indices)  
# 4. define train & testloaders, with dataset (1.) and subsamplers (3.), specify batch size
# 5. proceed with standard testing and training, by iterating through train- & test-loaders

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



# def batch_split(X, batch):
#   return (X[batch, :], np.delete(X, batch, 0))


# def k_fold(X, y, n_batches, kernel, lambda_reg):
#   losses = []
#   if n_batches > X.shape[0]: n_batches = X.shape[0]
#   batches = np.array_split(np.arange(X.shape[0]), n_batches)
#   loss = 0.
#   for count, batch in enumerate(batches):
#     X_test, X_train = batch_split(X, batch)
#     y_test, y_train = batch_split(y, batch)
#     krr = KRR(X_train, y_train, lambda_reg, kernel)
#     train_loss = krr.loss(y_train, krr.predict(X_train))
#     test_loss = krr.loss(y_test, krr.predict(X_test))
#     #print(f'run {count+1}: train loss {train_loss}, test loss {test_loss}')
#     loss += test_loss
#   #print(f'total score: {loss/n_batches}')
#   return loss/n_batches