# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 09:47:01 2019

@author: Thulitha
"""
import numpy as np

import pandas as pd
from pylab import rcParams
import seaborn as sb
import matplotlib.pyplot as plt

import sklearn
from sklearn.cluster import DBSCAN
from collections import Counter

#Setting standard data visualizing parameters
###############################################################################
rcParams['figure.figsize'] = 5,3
sb.set_style('whitegrid')

#Importing data
###############################################################################
dataset =pd.read_csv('Trades.csv')
stock = dataset[dataset['Stock'] == 'ES0158252033']
buy_target = stock.iloc[:,[1,-1,-4]].values

#Encoding Categorical data
###############################################################################
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
buy_target[:,1] = labelencoder.fit_transform(buy_target[:,1])

#Creating the DBSCAN model
###############################################################################
model = DBSCAN(eps=115, min_samples=3).fit(buy_target)

#Visualizing the outliers in results
###############################################################################
out = pd.DataFrame(buy_target)
print(Counter(model.labels_))
print(out[model.labels_ == -1])

fig = plt.figure()
ax = fig.add_axes([.1,.1,1,1])
colors=['red' if l == -1 else 'green' for l in model.labels_]
ax.scatter(buy_target[:,1], buy_target[:,0], c=colors, s=20)
ax.set_xlabel('Buy Broker')
ax.set_ylabel('Quantity')
plt.title('DBSCAN - With Buy Brokers')














