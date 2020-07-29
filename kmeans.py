# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:29:15 2020

@author: muadgra

K Means algorithm example with randomly generated data.

"""
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

X, y = make_blobs(n_samples = 300, cluster_std = 1.00, random_state = 8)
plt.scatter(X[:, 0], X[:, 1])
plt.show()

#Elbow method is used here.
#Algorithm runs several times and we aim to find the drastic drop.
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#Since the drastic drop is 3, we run the rest of the algorithm using 3.
kmeans = KMeans(n_clusters = 3)
pred_y = kmeans.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], cmap = 'viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'red')
