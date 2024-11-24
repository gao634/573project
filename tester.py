import knn
import ann
import rf
import dataprocessor as dp
import numpy as np
from sklearn.model_selection import KFold

def kfold(k, func, params, X, y):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    train_accs = []
    test_accs = []
    for train_index, test_index in kf.split(X):
        # Split data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        train_acc, test_acc = func(params, X_train, X_test, y_train, y_test)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

    return np.mean(train_accs), np.mean(test_accs)

#X, y = dp.label('obesity_min_max_scaled.csv')
X, y = dp.label('obesity_neural.csv')
#train_acc, test_acc = kfold(10, knn.KNN, 10, X, y)
#train_acc, test_acc = kfold(10, rf.RF, 5, X, y)
train_acc, test_acc = kfold(10, ann.ANN, None, X, y)
print(f"Average Training Accuracy: {train_acc:.2f}")
print(f"Average Testing Accuracy: {test_acc:.2f}")
