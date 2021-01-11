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
rcParams['figure.figsize'] = 13,6
sb.set_style('whitegrid')

#Importing data
###############################################################################
dataset =pd.read_csv('Trades.csv')
stock = dataset[dataset['Stock'] == 'ES0158252033']
buy_target = stock.iloc[:,[1,-1]].values

#Encoding Categorical data
###############################################################################
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
buy_target[:,1] = labelencoder.fit_transform(buy_target[:,1])

# Using the dendrogram to find the optimal number of clusters
###############################################################################
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(buy_target, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Trades')
plt.ylabel('Euclidean distances')
plt.show()

# Fitting Hierarchical Clustering to the dataset
###############################################################################
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(buy_target)

# Visualising the clusters
###############################################################################
plt.scatter(buy_target[y_hc == 0, 1], buy_target[y_hc == 0, 0], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(buy_target[y_hc == 1, 1], buy_target[y_hc == 1, 0], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(buy_target[y_hc == 2, 1], buy_target[y_hc == 2, 0], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(buy_target[y_hc == 3, 1], buy_target[y_hc == 3, 0], s = 100, c = 'cyan', label = 'Cluster 4')
plt.title('Clusters of trades')
plt.xlabel('Buy Brokers')
plt.ylabel('Trade Quantity')
plt.legend()
plt.show()














