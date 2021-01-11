# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 21:43:13 2019

@author: Thulitha
"""

import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sb

import pandas as pd
from apyori import apriori

#Setting standard data visualizing parameters
###############################################################################
rcParams['figure.figsize'] = 5,3
sb.set_style('whitegrid')

#Importing data
###############################################################################
dataset =pd.read_csv('Trades.csv')
stock = dataset[dataset['Stock'] == 'ES0158252033']
x=stock.iloc[:,[-2,-1]]

records = []
for i in range(0, 1980):
    records.append([str(x.values[i,j]) for j in range(0, 2)])

association_rules = apriori(records, min_support=0.001, min_confidence=0.3, min_lift=3, min_length=2)
association_results = list(association_rules)

#print(len(association_results))
#print(association_results[0])

for item in association_results:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")