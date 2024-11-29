import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

if torch.cuda.is_available():
    print(f"\nGPU: {torch.cuda.get_device_name(0)} is available")
else:
    print('\nNo GPU available')

device = torch.device("cuda")

writer = SummaryWriter()

pickle_folder = "pickle/"

with open(pickle_folder + "X_train.pickle", "rb") as f:
    X_train = pickle.load(f)

with open(pickle_folder + "Y_train.pickle", "rb") as f:
    Y_train = pickle.load(f)

with open(pickle_folder + "X_test.pickle", "rb") as f:
    X_test = pickle.load(f)

with open(pickle_folder + "Y_test.pickle", "rb") as f:
    Y_test = pickle.load(f)

X_train = torch.tensor(X_train).float().unsqueeze(1)
Y_train = torch.tensor(Y_train, dtype=torch.long)
X_test = torch.tensor(X_test).float().unsqueeze(1)
Y_test = torch.tensor(Y_test, dtype=torch.long)

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=512, kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(1152, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 4)  # Output layer
        
        self.testfc1 = nn.Linear(128, 4)
    def forward(self, x):
        x = (torch.relu(self.conv1(x)))
        x = self.maxpool(torch.relu(self.conv2(x)))
        x = (torch.relu(self.conv3(x)))
        x = self.maxpool(torch.relu(self.conv4(x)))
        x = (torch.relu(self.conv5(x)))
        x = x.view(x.size(0), -1)
        #print(f"Flattened layer size: {x.shape}")   # use this to adjust the number of input channels going into the first linear layer
        x = self.testfc1(x)
        return x

model = NeuralNet()
# Create dataset and dataloaders
train_dataset = TensorDataset(X_train, Y_train.float())
test_dataset = TensorDataset(X_test, Y_test.float())

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

learning_rate = 0.001

model = NeuralNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Function to calculate accuracy
def calculate_accuracy(y_pred, y_true):
    _, true = torch.max(y_true, 1)
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == true).sum().item()
    accuracy = correct / y_true.size(0)
    return accuracy

# Training loop
num_epochs = 75
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    
    for batch_X, batch_Y in train_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_accuracy += calculate_accuracy(outputs, batch_Y)
    
    epoch_loss /= len(train_loader)
    epoch_accuracy /= len(X_train)
    writer.add_scalar("Loss/epoch with learning rate: {learning_rate}", epoch_loss, epoch)
    writer.add_scalar("Accuracy/epoch  with learning rate: {learning_rate}", epoch_accuracy, epoch)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy * 100:.2f}%')

writer.close()

# Testing loop
model.eval()
test_accuracy = 0.0

with torch.no_grad():
    for batch_X, batch_Y in test_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        outputs = model(batch_X)        
        test_accuracy += calculate_accuracy(outputs, batch_Y)

test_accuracy /= len(X_test)
writer.add_scalar(f"Test Accuracy/epoch with learning rate: {learning_rate}", test_accuracy, epoch)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
writer.close()