# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 21:06:18 2020

@author: muadgra

Logistic Regression from scratch.
"""

import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate = 0.001,
                 n_iterations = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        #gradient descent
        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_classes = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_classes
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    

from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

regressor = LogisticRegression(learning_rate = 0.0001, n_iterations = 1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
print(accuracy(y_test, predictions))
