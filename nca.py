import numpy as np
import pandas as pd
import dataprocessor as dp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from metric_learn import NCA as NCA_learn
from sklearn.pipeline import make_pipeline

def NCA(k, nca, X_train, X_test, y_train, y_test):
    # Initialize NCA and fit it
    # Transform both training and test sets
    X_train_transformed = nca.transform(X_train)
    X_test_transformed = nca.transform(X_test)
    
    # Initialize kNN and train it on the transformed training data
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_transformed, y_train)
    
    # Predict on the transformed test data
    y_pred = knn.predict(X_test_transformed)
    train_pred = knn.predict(X_train_transformed)
    
    # Calculate accuracy
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    return train_accuracy, test_accuracy