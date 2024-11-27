import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
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
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.fc1 = nn.Linear(21760, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 4)  # Output layer

    def forward(self, x):
        x = self.maxpool(torch.relu(self.conv1(x)))
        x = self.maxpool(torch.relu(self.conv2(x)))
        x = self.maxpool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        # print(f"Flattened layer size: {x.shape}")   # use this to adjust the number of input channels going into the first linear layer
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)  # Output layer
        return x


model = NeuralNet()
# Create dataset and dataloaders
train_dataset = TensorDataset(X_train, Y_train.float())
test_dataset = TensorDataset(X_test, Y_test.float())

bs = 2
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

learning_rate = 0.001

model = NeuralNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Print model summary
print(model)
print(optimizer)
print(criterion)
print(f"Batch Size: {bs}")

# Create a trial file
def create_trial_file(trials_folder="trials/"):
    trial_number = 1
    while os.path.exists(trials_folder + f"trial_{trial_number}.txt"):
        trial_number += 1
    return trials_folder + f"trial_{trial_number}.txt"


trial_file = create_trial_file()

with open(trial_file, "w") as f:
    f.write(f"Model architecture: {model}\n")
    f.write(f"Optimizer: {optimizer}\n")
    f.write(f"Criterion: {criterion}\n")
f.close()

# Function to calculate accuracy


def calculate_accuracy(y_pred, y_true):
    _, true = torch.max(y_true, 1)
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == true).sum().item()
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
        epoch_accuracy += calculate_accuracy(outputs, batch_Y)

    epoch_loss /= len(train_loader)
    epoch_accuracy /= len(X_train)
    writer.add_scalar(
        "Loss/epoch with learning rate: {learning_rate}", epoch_loss, epoch)
    writer.add_scalar(
        "Accuracy/epoch  with learning rate: {learning_rate}", epoch_accuracy, epoch)
    print(
        f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy * 100:.2f}%')
    
    # Write to trial file
    with open(trial_file, "a") as f:
        f.write(
            f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy * 100:.2f}%\n')
    f.close()


writer.close()

# Testing loop
model.eval()
test_accuracy = 0.0

with torch.no_grad():
    for batch_X, batch_Y in test_loader:
        outputs = model(batch_X)
        test_accuracy += calculate_accuracy(outputs, batch_Y)

test_accuracy /= len(X_test)
writer.add_scalar(
    f"Test Accuracy/epoch with learning rate: {learning_rate}", test_accuracy, epoch)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Write to trial file
with open(trial_file, "a") as f:
    f.write(f'Test Accuracy: {test_accuracy * 100:.2f}%')
f.close()

writer.close()
