# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 09:23:03 2019

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
from sklearn.cluster import KMeans

#Setting standard data visualizing parameters
###############################################################################
rcParams['figure.figsize'] = 8,6
sb.set_style('whitegrid')

#Importing data
###############################################################################
dataset =pd.read_csv('Trades.csv')
stock = dataset[dataset['Stock'] == 'ES0158252033']
buy_target = stock.iloc[:,[-1,1]].values

#Encoding Categorical data
###############################################################################
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
buy_target[:,0] = labelencoder.fit_transform(buy_target[:,0])
classes = labelencoder.classes_
length = np.size(classes, axis=0)

#Creating the DBSCAN model
###############################################################################
labels_ = []
targets_ = []
for i in range(0,length):
    target = buy_target[buy_target[:,0] == i]
    
    #Kmeans attribute calculation
    ###########################################################################
#    kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 40)
#    kmeans.fit(target)
#    sm = np.size(target, axis=0)*0.315
#    eps_ = np.sqrt(kmeans.inertia_/sm)
    
    #standard deviation attribute calculation
    ###########################################################################
    n=1.45
    eps_ = n*np.std(target[:,1])
    sm = np.size(target, axis=0)/n
    ###########################################################################
        
    model = DBSCAN(eps=eps_, min_samples=sm, algorithm='ball_tree').fit(target)
    targets_.append(target)
    labels_.append(model.labels_)
    
labels = np.concatenate(labels_, axis=0)
target = np.concatenate(targets_, axis=0)

#Visualizing the outliers in results
###############################################################################
out = pd.DataFrame(target)
print(Counter(labels))
#print(out[labels == -1])

fig = plt.figure()
ax = fig.add_axes([.1,.1,1,1])
colors=['red' if l == -1 else 'green' for l in labels]
ax.scatter(target[:,0], target[:,1], c=colors, s=20)
ax.set_xlabel('Buy Broker')
ax.set_ylabel('Quantity')
plt.title('DBSCAN - With Buy Brokers')