# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 15:51:41 2020

@author: mertc

Logistic Regression applied to Titanic Data Set
"""

#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import math
#import seaborn as sns

titanic_data = pd.read_csv('./titanic/train.csv')

#sns.countplot(x = 'Survived', data = titanic_data)

#Removing of NaN values and unnecessary columns.
titanic_data.drop("Cabin", axis = 1, inplace = True)
titanic_data.dropna(inplace = True)
sex = pd.get_dummies(titanic_data['Sex'], drop_first = True)
embark = pd.get_dummies(titanic_data['Embarked'], drop_first = True)
p_class = pd.get_dummies(titanic_data["Pclass"], drop_first = True)

titanic_data = pd.concat([titanic_data, sex, embark, p_class], axis = 1)
titanic_data.drop(['Sex', 'Embarked', 'PassengerId', 'Name', 'Ticket', 'Pclass'], axis = 1, inplace = True)

#Train Phase

X = titanic_data.drop("Survived", axis = 1)
y = titanic_data["Survived"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report
#print(classification_report(y_test, predictions))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, predictions))

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))
