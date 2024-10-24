import numpy as np
import pandas as pd
import dataprocessor as dp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
def RF(n_estimators, X_train=None, X_test=None, y_train=None, y_test=None):
    #X, y = dp.label('obesity.csv')
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    train_pred = rf.predict(X_train)
    
    #print("Accuracy test:", accuracy_score(y_test, y_pred))
    return accuracy_score(y_train, train_pred), accuracy_score(y_test, y_pred)
#RF(2)