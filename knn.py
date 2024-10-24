import numpy as np
import pandas as pd
import dataprocessor as dp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def KNN(k, X_train, X_test, y_train, y_test):
    #X, y = dp.label()
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    train_pred = knn.predict(X_train)

    #print("Accuracy:", accuracy_score(y_test, y_pred))
    return accuracy_score(y_train, train_pred), accuracy_score(y_test, y_pred)
#KNN()