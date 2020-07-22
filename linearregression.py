# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:52:00 2020

@author: muadgra 

Linear Regression used with Head Brain Data Set
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (20.0, 10.0)

#Reading data
data = pd.read_csv('headbrain.csv')
#print(data.head())
X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values

mean_x = np.mean(X)
mean_y = np.mean(Y)

m = len(X)

#b1 and b2 calculations

numer = 0
denom = 0

for i in range(m):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2

b1 = numer / denom
b0 = mean_y - (b1 * mean_x)

max_x = np.max(X) + 100
min_x = np.max(X) - 100

x = np.linspace(min_x, max_x, 1000)
y = b0 + b1 * x

'''
plt.plot(x, y, color = '#58b970', label = 'Regression Line')
plt.scatter(X, Y, c = '#ef5423', label = 'Scatter Plot')
plt.xlabel('Head size in cm^3')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()
'''

ss_t = 0
ss_r = 0
y_pred = 0

for i in range(m):
    y_pred = b0 + b1 * X[i]
    ss_t += (Y[i] - mean_y) ** 2
    ss_r += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_r / ss_t)
print(r2)
