# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 09:17:07 2020

@author: Tan Phan - phanttan@gmail.com
"""

"""
Try to build a classifier for the MNIST dataset that achieves over 97% accuracy
on the test set. Hint: the KNeighborsClassifier works quite well for this task;
you just need to find good hyperparameter values (try a grid search on the
weights and n_neighbors hyperparameters).
"""

# Get Data
import numpy as np
try:
    # Python 2
    from urllib2 import HTTPError
except ImportError:
    # Python 3
    from urllib.error import HTTPError

try:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    mnist.target = mnist.target.astype(np.int8) # fetch_openml() returns targets as strings
    reorder_train = np.array(sorted([(target, i) for i, target in 
                             enumerate(mnist.target[:60000])]))
    reorder_test = np.array(sorted([target, i]) for i, target in 
                            enumerate(mnist.target[60000:]))
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_test]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.data[reorder_test + 60000]
except ImportError:
      
    from scipy.io import loadmat
    from six.moves.urllib.request import urlopen
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    mnist_path = "./mnist-original.mat"
    try:
        mldata_url = urlopen(mnist_alternative_url)
    except HTTPError as e:
        if e.code == 404:
            e.message = "Dataset MNIST not found on github"
        raise
    # load dataset matlab file
    with open(mnist_path, "wb") as f:
        content = mldata_url.read()
        f.write(content)
    mnist_raw = loadmat(mnist_path)
    mnist = {
        "data": mnist_raw["data"].T,
        "target": mnist_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "mldata.org dataset: mnist-original",
            }
# Prepare the data
X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train,y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_idx = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_idx], y_train[shuffle_idx]

# Train a model using K-Neighbor Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
param_grid = [{'weights':["uniform","distance"],'n_neighbors':[3,4,5]}]
knn_clf = KNeighborsClassifier()
# Using Grid Search Cross Validation
grid_search = GridSearchCV(knn_clf, param_grid, cv=5, verbose=3, n_jobs = -1)
grid_search.fit(X_train, y_train)
grid_search.best_params_
grid_search.best_scores_
# Visualization the Grid Search
cvres = grid_search.cv_results_

# Validation
from sklearn.metrics import accuracy_score 
y_pred = grid_search.predict(X_test)
res = accuracy_score(y_test, y_pred)