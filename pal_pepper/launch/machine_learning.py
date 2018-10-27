#!/usr/bin/env python

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

plt.subplot(3, 2, 1)
x1 = np.array([1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
x2 = np.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])
X = np.array(zip(x1, x2)).reshape(len(x1), 2)

plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title('Instances')
plt.scatter(x1, x2)

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
markers = ['o', 's', 'D', 'v', '^', 'p', '*', '+']

clusters = [2, 3, 4, 5, 8]
subplot_counter = 1
sc_scores = []
for t in clusters:
    subplot_counter += 1
    plt.subplot(3, 2, subplot_counter)
    kmean_mode1 = KMeans(n_clusters=t).fit(X)

    for i, l in enumerate(kmean_mode1.labels_):
        plt.plot(x1[i], x2[i], colors=colors[l], marker=markers[l], ls='None')

    plt.xlim([0, 10])
    plt.ylim([0, 10])
    sc_score = silhouette_score(X, kmean_mode1.labels_, metric='euclidean')
    sc_scores.append(sc_score)
    plt.title('K=%s, silhouette coeddicient=%0.03f' %(t, sc_score))
plt.figure()
plt.plot(clusters, sc_scores, '*-')
plt.xlabel('Number of Clusters')
plt.ylabel("Silhouette Coefficient Score")

plt.show()

'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics

digits_train = pd.read_csv("https://archive.ics.uci.edu/ml/"
                           "machine-learning-databases/optdigits/optdigits.tra", header=None)
digits_test = pd.read_csv("https://archive.ics.uci.edu/ml/"
                           "machine-learning-databases/optdigits/optdigits.tes", header=None)
X_train = digits_train[np.arange(64)]
y_train = digits_train[64]

X_test = digits_test[np.arange(64)]
y_test = digits_test[64]

kmeans = KMeans(n_clusters=10)
kmeans.fit(X_train)
y_pre = kmeans.predict(X_test)

print metrics.adjusted_rand_score(y_test, y_pre)
'''








