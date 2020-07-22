# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 15:18:01 2020

@author: mertc

Linear Regression of Head Brain Data Set, with Sci Kit Library
"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Reading data
data = pd.read_csv('headbrain.csv')
X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values
m = len(X)

#In SciKit learn, we can'T use Rank 1 matrixes
X = X.reshape((m, 1))

reg = LinearRegression()

reg = reg.fit(X, Y)
Y_pred = reg.predict(X)

r2_score = reg.score(X, Y)
