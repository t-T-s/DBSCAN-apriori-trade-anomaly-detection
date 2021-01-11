# DBSCAN-apriori-trade-anomaly-detection
This repository contains an implementation of DBSCAN and Apriori algorithms to identify potential anomalies in stock market trading data.

The project outline is as follows:

Stock markets facilitate trading of company stocks among market participants (i.e., traders) at an agreed price. Market participants are allowed to submit their buy/sell interests (i.e., orders) to the stock market and the stock market matches these buy/sell orders based on their interested buying/selling price (simplest scenario). The given Trades.csv file contains a set of trades done by multiple traders on multiple stocks in a stock market. 

1. Use suitable unsupervised learning techniques with necessary justifications to identify both outlier trades (i.e., traded quantities) and traders in one selected stock in the given dataset.
2. Also explain and implement an approach that can be used to identify collusive trader groups in the full dataset.

DBSCAN groups together points that are close to each other based on a distance measurement (usually Euclidean distance) and a minimum number of points. It also marks as outliers the points that are in low-density regions.

The second part of the project made me think of so many ways to tackle this problem. But due to the time constraint finally I had to limit myself to only one approach. However it can be used to get a rough understanding about the relationship between brokers who work as a *collusive trader group*. Despite of that I tried on numerous other methods. But due to lack of time a complete implementation was not possible.

Apriori algorithm is used to identify relationship between items purchased by people in the markets. Here the algorithm considers three main parameters between different combinations of the same items as *support, confidence and lift.* In this scenario it helps to understand the reverse relation between brockers where same stocks are purchased.
