# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 10:57:41 2021

@author: Mertcan
"""
import numpy as np
import pandas as  pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Train Test Split
from sklearn.model_selection import train_test_split

# Models
from sklearn.svm import SVC

# Metrics
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV
heart_data = pd.read_csv('C:/Users/Mertcan/Desktop/data/heart.csv')
categorical_columns = ['sex', 'exng','caa','cp','fbs','restecg','slp','thall']
continuous_columns = ['age', 'trtbps','chol','thalachh','oldpeak']
df = pd.get_dummies(heart_data, columns = categorical_columns, drop_first =True)
X = df.drop(['output'], axis = 1)
y = df[['output']]
scaler = RobustScaler()

X[continuous_columns] = scaler.fit_transform(X[continuous_columns])
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 8)

svm = SVC(probability=True)
parameters = {"C":np.arange(1,10,1),'gamma':[0.001,0.005,0.01,0.05,0.1,0.5,1,5]}

# instantiating the GridSearchCV object
searcher = GridSearchCV(svm, parameters)

# fitting the object
searcher.fit(X_train, y_train.values.ravel())

# predicting the values
y_pred = searcher.predict(X_test)

# printing the test accuracy
print("Best score after hyperparameter tuning: {0} - \n Best score before hyperparameter tuning: {1}".format(accuracy_score(y_test, y_pred), searcher.best_score_))

print(confusion_matrix(y_test,y_pred))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)

#Remove comment to display ROC
'''
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for Heart disease classifier')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)
'''