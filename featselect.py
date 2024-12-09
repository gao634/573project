import numpy as np
import pandas as pd
import dataprocessor as dp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def main():
    X, y = dp.label('obesity.csv')
    rf = RandomForestClassifier(n_estimators=17, random_state=42)
    rf.fit(X, y)
    importances = np.asarray(rf.feature_importances_)
    print(importances)

if __name__ == '__main__':
    main()