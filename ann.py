import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



class NeuralNetwork(nn.Module):
    def __init__(self, input):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 8)
        self.fc5 = nn.Linear(8, 4)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        #x = self.softmax(self.fc5(x))
        return x
    
def trainer(par, X_train, y_train):
    if par is None:
        epochs = 200
        lr = 0.001
        batch = 50
        ver = 1
    else: epochs, lr, batch, ver = par
    
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)

    model = NeuralNetwork(X_train.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_epochs = epochs
    losses = []
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
            plt.savefig(f'losses/ann/loss{ver}.png')
    torch.save(model, f'losses/models/ann{ver}.pth')
    return model

def tester(model, X_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    correct = 0
    total = 0

    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_tensor = torch.tensor(y_test, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=200, shuffle=True)
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)

            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    #print(f'NN acc: {accuracy:.2f}%')
    return accuracy

def ANN(par, X_train, X_test, y_train, y_test):
    X_train = X_train.values.astype(int)
    X_test = X_test.values.astype(int)
    y_train = y_train.values.astype(int)
    y_test = y_test.values.astype(int)
    #print(X_train.dtype)

    model = trainer(par, X_train, y_train)
    accuracy_train = tester(model, X_train, y_train)
    accuracy_test = tester(model, X_test, y_test)
    return accuracy_train, accuracy_test