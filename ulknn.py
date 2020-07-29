# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:40:36 2020

@author: mertc

KNN used on IRIS dataset.

"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 8)
scores = {}
scores_list = []
krange = range(1, 26)

for k in krange:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    scores[k] = metrics.accuracy_score(y_test, y_pred)
    scores_list.append(metrics.accuracy_score(y_test, y_pred))

plt.plot(krange, scores_list)
plt.xlabel('Values of K')
plt.ylabel('Accuracy of Test')
plt.show()

