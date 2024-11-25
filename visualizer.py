import tester
import knn
import matplotlib.pyplot as plt
import dataprocessor as dp
from sklearn.model_selection import train_test_split
import rf
import numpy as np
import nca
import warnings
from metric_learn import NCA as NCA_learn


warnings.filterwarnings('ignore')
def plot(func, X, y, path):
    k_values = range(1, 21)
    training_accuracies = []
    testing_accuracies = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ncaa = NCA_learn(max_iter=100, random_state=42)
    ncaa.fit(X_train, y_train)

    for k in k_values:
        print(k)
        #train_acc, test_acc = tester.kfold(10, func, k, X, y)
        train_acc, test_acc = func(k, ncaa, X_train, X_test, y_train, y_test)
        training_accuracies.append(train_acc)
        testing_accuracies.append(test_acc)

    # Find highest test acc
    max_test_acc = max(testing_accuracies)
    max_k = k_values[np.argmax(testing_accuracies)]

    print("Highest Test Accuracy: {:.2f}% at k={}".format(max_test_acc * 100, max_k))

    plt.figure(figsize=(10, 5))
    plt.plot(k_values, training_accuracies, label='Training Accuracy')
    plt.plot(k_values, testing_accuracies, label='Testing Accuracy', marker='o')
    
    plt.scatter(max_k, max_test_acc, color='red')  # mark the highest point
    plt.annotate(f'Highest Acc: {max_test_acc:.2f}',  # this is the text
                 (max_k, max_test_acc),  # this is the point to label
                 textcoords="offset points",  # how to position the text
                 xytext=(0,10),  # distance from text to points (x,y)
                 ha='center')  # horizontal alignment can be left, right or center

    plt.title('K Nearest Neighbors with NCA')
    plt.xlabel('Number of Neighbors')
    #plt.title('Random Forest Training and Testing Accuracies')
    #plt.xlabel('Number of Decison Trees')
    plt.ylabel('Accuracy')
    plt.xticks(k_values)
    plt.legend()
    plt.grid(True)
    plt.savefig(path, format='png', dpi=300)
    plt.show()

X, y = dp.label('obesity_encoded.csv')
#X, y = dp.label('obesity_min_max_scaled.csv')
#X, y = dp.label('obesity.csv')
path = 'figures/nca.png'
plot(nca.NCA, X, y, path)