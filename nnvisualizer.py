import dataprocessor as dp
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ann import NeuralNetwork, tester
def main():
    X, y = dp.label('obesity_neural.csv')
    X = X.values.astype(int)
    y = y.values.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = 0.0005
    batch = 100
    ver = 21


    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)

    model = NeuralNetwork(X_train.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_epochs = 2000
    losses = []
    train_accs = []
    test_accs = []
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Update plot every 5 epochs
        if (epoch + 1) % 5 == 0:
            plt.clf() 
            plt.plot(range(1, epoch+2), losses, label='Training Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.legend()
            plt.savefig(f'training/ann/losses/loss{ver}.png')
        # test accuracy every 100 epochs
        if (epoch + 1) % 100 == 0:
            accuracy_train = tester(model, X_train, y_train)
            accuracy_test = tester(model, X_test, y_test)
            train_accs.append(accuracy_train)
            test_accs.append(accuracy_test)
            torch.save(model, f'training/ann/models/ann{ver}ep{epoch+1}.pth')
            model.train()

    x_ticks = (np.asarray(range(len(train_accs))) +1) * 100
    max_test_acc = max(test_accs)
    max_k = x_ticks[np.argmax(test_accs)]

    print("Highest Test Accuracy: {:.2f}% at k={}".format(max_test_acc * 100, max_k))

    plt.figure(figsize=(10, 5))
    plt.plot(x_ticks, train_accs, label='Training Accuracy')
    plt.plot(x_ticks, test_accs, label='Testing Accuracy', marker='o')
    
    plt.scatter(max_k, max_test_acc, color='red')  
    plt.annotate(f'Highest Acc: {max_test_acc:.2f}',  
                 (max_k, max_test_acc),  # this is the point to label
                 textcoords="offset points",  # how to position the text
                 xytext=(0,10),  # distance from text to points (x,y)
                 ha='center')  # horizontal alignment can be left, right or center

    plt.title('ANN Accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xticks(x_ticks)
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/ann.png', format='png', dpi=300)
    plt.show()
    return model

if __name__ == '__main__':
    main()