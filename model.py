import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, TensorDataset

# Load data from pickle files
pickle_folder = "pickle/"

with open(pickle_folder + "X_train.pickle", "rb") as f:
    X_train = pickle.load(f)

with open(pickle_folder + "Y_train.pickle", "rb") as f:
    Y_train = pickle.load(f)

with open(pickle_folder + "X_test.pickle", "rb") as f:
    X_test = pickle.load(f)

with open(pickle_folder + "Y_test.pickle", "rb") as f:
    Y_test = pickle.load(f)

# Convert data to PyTorch tensors and add channel dimension
X_train = torch.tensor(X_train).float().unsqueeze(1)
Y_train = torch.tensor(Y_train, dtype=torch.long)
X_test = torch.tensor(X_test).float().unsqueeze(1)
Y_test = torch.tensor(Y_test, dtype=torch.long)

print(X_train.shape)
print(X_test.shape)
print('Y_train data type: ', Y_train.dtype)
print('Y_train data type: ', Y_test.dtype)

# Define the neural network
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.fc1 = nn.Linear(21760, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 4)  # Output layer

    def forward(self, x):
        x = self.maxpool(torch.relu(self.conv1(x)))
        x = self.maxpool(torch.relu(self.conv2(x)))
        x = self.maxpool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        print(f"Flattened layer size: {x.shape}")
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)  # Output layer
        return x

# Create dataset and dataloaders
train_dataset = TensorDataset(X_train, Y_train.long())
test_dataset = TensorDataset(X_test, Y_test.long())

batch_size = 32  # Adjust based on memory constraints
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
model = NeuralNet()
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
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_accuracy += (outputs.argmax(1) == batch_Y).sum().item()
    
    epoch_loss /= len(train_loader)
    epoch_accuracy /= len(X_train)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy * 100:.2f}%')

# Testing loop
model.eval()
test_loss = 0.0
test_accuracy = 0.0

with torch.no_grad():
    for batch_X, batch_Y in test_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        
        test_loss += loss.item()
        test_accuracy += (outputs.argmax(1) == batch_Y).sum().item()

test_loss /= len(test_loader)
test_accuracy /= len(X_test)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%')
