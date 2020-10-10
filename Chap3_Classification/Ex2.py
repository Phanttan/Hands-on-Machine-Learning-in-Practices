# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 11:10:18 2020

@author: Tan Phan - phanttan@gmail.com
"""

"""
Write a function that can shift an MNIST image in any direction (left, right, up,
or down) by one pixel.5 Then, for each image in the training set, create four shif‐
ted copies (one per direction) and add them to the training set. Finally, train your
best model on this expanded training set and measure its accuracy on the test set.
You should observe that your model performs even better now! This technique of 
artificially growing the training set is called data augmentation or training set
expansion.
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

from scipy.ndimage.interpolation import shift

def shift_image(image, dx, dy):
    image = image.reshape((28,28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])

image = X_train[1000]
shifted_image_down = shift_image(image, 0, 5)
shifted_image_left = shift_image(image, -5, 0)

# Data Augmentation
X_train_augmented = [image for image in X_train]
y_train_augmented = [image for image in y_train]

for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    for image, label in zip(X_train, y_train):
        X_train_augmented.append(shift_image(image, dx, dy))
        y_train_augmented.append(label)

X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

shuffle_idx = np.random.permutation(len(X_train_augmented))
X_train_augmented = X_train_augmented[shuffle_idx]

# Train a model using K-Neighbor Classifier
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
# Using Grid Search Cross Validation
from sklearn.model_selection import GridSearchCV
param_grid = [{'weights':["uniform","distance"],'n_neighbors':[3,4,5]}]
grid_search = GridSearchCV(knn_clf, param_grid, cv= 5, verbose=3, n_jobs= -1)
grid_search.fit(X_train_augmented, y_train_augmented)
# Train again with best model
knn_clf = KNeighborsClassifier(**grid_search.best_params_)
from sklearn.metrics import accuracy_score
y_pred = knn_clf.predict(X_test)
res = accuracy_score(y_test, y_pred)