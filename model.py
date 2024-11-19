import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os


with open("X_train.pickle", "rb") as f:
    X_train = pickle.load(f)

with open("Y_train.pickle", "rb") as f:
    Y_train = pickle.load(f)

with open("X_test.pickle", "rb") as f:
    X_test = pickle.load(f)

with open("Y_test.pickle", "rb") as f:
    Y_test = pickle.load(f)

X_train = torch.tensor(X_train)
Y_train = torch.tensor(Y_train)
X_test = torch.tensor(X_test)
Y_test = torch.tensor(Y_test)
X_train = X_train.unsqueeze(1)
X_test = X_test.unsqueeze(1)
print(X_train.shape)
print(X_test.shape)

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=2, padding=1)   # in_channels is set to 1 since the images are grayscale. Keep that in mind in case we want to do RGB images instead
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.fc1 = nn.Linear(21760, 512)
        self.fc2 = nn.Linear(512, 256)  
        self.fc3 = nn.Linear(256, 128)  
        self.fc4 = nn.Linear(128, 4) # output layer
    def forward(self, x):
        x = self.maxpool(torch.relu(self.conv1(x)))
        x = self.maxpool(torch.relu(self.conv2(x)))
        x = self.maxpool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x) # output layer
        return x
    
model = NeuralNet()

print(len(X_train), "\n")
print(len(Y_train), "\n")
print(len(X_test), "\n")
print(len(Y_test), "\n")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Function to calculate accuracy
def calculate_accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y_true).sum().item()
    accuracy = correct / y_true.size(0)
    return accuracy

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train)
    loss = criterion(outputs, Y_train)
    
    loss.backward()
    optimizer.step()
    
    accuracy = calculate_accuracy(outputs, Y_train)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy * 100:.2f}%')

print("\n\n")

# Testing loop
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, Y_test)
    test_accuracy = calculate_accuracy(test_outputs, Y_test)
    print(f'Test Loss: {test_loss.item():.4f}, Test Accuracy: {test_accuracy * 100:.2f}%')
