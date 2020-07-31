# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 01:17:34 2020

@author: muadgra

Support Vector Machine used on digit dataset to classify digits.

"""
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()
clf = svm.SVC(gamma = 0.001,
              C= 100)
#store all of the data and answers
x, y = digits.data[:-1], digits.target[:-1]

clf.fit(x, y)
#printing prediction of last digit, and it's actual value
print(clf.predict(digits.data[-1].reshape(1, -1)))
print(digits.target[-1])
#print(clf.predict((digits.data))[-1])
