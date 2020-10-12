# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 12:13:12 2020

@author: Tan Phan - phanttan@gmail.com
"""

import pandas as pd
import os
    ## Get the data ##
TITANIC_PATH = os.path.join("datasets", "titanic")

def load_titanic_data(filename, titanic_path= TITANIC_PATH):
    csv_path = os.path.join(titanic_path, filename)
    return pd.read_csv(csv_path)
    
train_data = load_titanic_data("train.csv")
test_data = load_titanic_data("test.csv")


    ## Preprocessing Data ## 
from sklearn.base import BaseEstimator, TransformerMixin

"""
A Class to select numerical or categorical columns since Scikit-learn does not 
handle DataFrames yet
"""
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]
    
from sklearn.pipeline import Pipeline
try:
    from sklearn.impute import SimpleImputer
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer
    
num_pipeline = Pipeline([ 
        ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch","Fare"])),
        ("imputer", SimpleImputer(strategy="median")),
        ]) #numerical pipeline
num_pipeline.fit_transform(train_data)
# SimpleImputer is not working for String, use MostFrequentImputer below to substitute
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series(
                [X[c].value_counts().index[0] for c in X], index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)
        
try:
    from sklearn.preprocessing import OneHotEncoder
except:
    from future_encoders import OneHotEncoder # Scikit-learn <0.20   
    
# Build Pipeline for the Categorical attributes:
cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(["Pclass","Sex","Embarked"])),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse=False)),
        ]) # categorical pipeline
cat_pipeline.fit_transform(train_data)

# Finally, join the numerical and categorical pipelines

from sklearn.pipeline import FeatureUnion
preprocess_pipeline = FeatureUnion(
        transformer_list=[("num_pipe",num_pipeline),("cat_pipe",cat_pipeline)])

X_train = preprocess_pipeline.fit_transform(train_data)
y_train = train_data["Survived"]

    ##Training Model##
# Using SVC
from sklearn.svm import SVC

svm_clf = SVC(gamma="auto")
svm_clf.fit(X_train, y_train)
X_test = preprocess_pipeline.transform(test_data)
y_pred = svm_clf.predict(X_test)
    ##Evaluation##
from sklearn.model_selection import cross_val_score
svm_scores_dict = dict()
for idx in range(10,15):
    svm_score = cross_val_score(svm_clf, X_train, y_train, cv=idx).mean()
    svm_scores_dict.update({idx:svm_score})
        # The best score is at cv=13, but it is not much enhanced
svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=13)
# Using RandomForest Classifier
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=13)
        # The score is better to SVC with 0.81

    ##Visualize the comparation results##
import matplotlib.pyplot as plt
plt.figure(figsize=(8,4))
plt.plot([1]*13, svm_scores, ".")
plt.plot([2]*13, forest_scores, ".")
plt.boxplot([svm_scores, forest_scores], labels=("SVM","RandomForest"))
plt.ylabel("Accuracy", fontsize=14)
plt.show()

