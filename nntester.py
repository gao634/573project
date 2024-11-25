import ann
import dataprocessor as dp
import numpy as np
from sklearn.model_selection import KFold, train_test_split

def main():
    X, y = dp.label('obesity_neural.csv')
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    lrs = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    batches = [50, 100, 200]
    ver = 0
    train_accs = []
    test_accs = []
    for lr in lrs:
        for batch in batches:
            print(ver)
            params = [200, lr, batch, ver]
            train_acc, test_acc = ann.ANN(params, X_train, X_val, y_train, y_val)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            ver += 1
    best_ver = np.argmax(np.asarray(test_accs))
    lr_ind = int(best_ver / 3)
    batch_ind = int(best_ver % 3)
    print(lrs[lr_ind], batches[batch_ind])
    #params = [2000, lrs[lr_ind], batches[batch_ind], ver]
    #train_acc, test_acc = ann.ANN(params, X_train, X_test, y_train, y_test)
    #print(f"Average Training Accuracy: {train_acc:.2f}")
    #print(f"Average Testing Accuracy: {test_acc:.2f}")

if __name__ == '__main__':
    main()