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
rcParams['figure.figsize'] = 5, 4
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

# Using the elbow method to find the optimal number of clusters
###############################################################################
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(buy_target)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#Creating the Kmeans model with 5 clusters
###############################################################################
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(buy_target)


# Visualising the clusters
###############################################################################
plt.scatter(buy_target[y_kmeans == 0, 1], buy_target[y_kmeans == 0, 0], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(buy_target[y_kmeans == 1, 1], buy_target[y_kmeans == 1, 0], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(buy_target[y_kmeans == 2, 1], buy_target[y_kmeans == 2, 0], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(buy_target[y_kmeans == 3, 1], buy_target[y_kmeans == 3, 0], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(buy_target[y_kmeans == 4, 1], buy_target[y_kmeans == 4, 0], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of trades')
plt.xlabel('Buy Broker ID')
plt.ylabel('Trade Quantities')
plt.legend()
plt.show()

#Creating the Kmeans model with 3 clusters
###############################################################################
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(buy_target)


# Visualising the clusters
###############################################################################
plt.scatter(buy_target[y_kmeans == 0, 1], buy_target[y_kmeans == 0, 0], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(buy_target[y_kmeans == 1, 1], buy_target[y_kmeans == 1, 0], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(buy_target[y_kmeans == 2, 1], buy_target[y_kmeans == 2, 0], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of trades')
plt.xlabel('Buy Broker ID')
plt.ylabel('Trade Quantities')
plt.legend()
plt.show()














