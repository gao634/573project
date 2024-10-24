import tester
import knn
import matplotlib.pyplot as plt
import dataprocessor as dp
import rf
import numpy as np
def plot(func, X, y, path):
    k_values = range(1, 21)
    training_accuracies = []
    testing_accuracies = []

    for k in k_values:
        train_acc, test_acc = tester.kfold(10, func, k, X, y)
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

    plt.title('K Nearest Neighbors Training and Testing Accuracies')
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
path = 'figures/knn3.png'
plot(knn.KNN, X, y, path)