import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

if torch.cuda.is_available():
    print(f"\nGPU: {torch.cuda.get_device_name(0)} is available")
else:
    print('\nNo GPU available')

device = torch.device("cuda")

writer = SummaryWriter()

pickle_folder = "meningioma-pickle/"

# Load pickled datasets
with open(pickle_folder + "X_train.pickle", "rb") as f:
    X_train = pickle.load(f)

with open(pickle_folder + "Y_train.pickle", "rb") as f:
    Y_train = pickle.load(f)

with open(pickle_folder + "X_test.pickle", "rb") as f:
    X_test = pickle.load(f)

with open(pickle_folder + "Y_test.pickle", "rb") as f:
    Y_test = pickle.load(f)

# Prepare data as tensors
X_train = torch.tensor(X_train).float().unsqueeze(1)  # Add channel dimension
# Ensure long type for classification
Y_train = torch.tensor(Y_train, dtype=torch.long)
X_test = torch.tensor(X_test).float().unsqueeze(1)
Y_test = torch.tensor(Y_test, dtype=torch.long)

# Define the neural network
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        # Global average pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 4)  # Output layer for 4 classes

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)  # Flatten for fully connected layers

        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x


model = NeuralNet().to(device)

# Create datasets and dataloaders
train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_test, Y_test)

bs = 96
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

learning_rate = 0.0001
# This loss function works for multi-class classification
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Log model details
print(model)
print(optimizer)
print(criterion)
print(f"Batch Size: {bs}")

# Create a trial file for logs
def create_trial_file(trials_folder="trials/four-class-model2"):
    trial_number = 1
    while os.path.exists(trials_folder + f"/trial_{trial_number}.txt"):
        trial_number += 1
    return trials_folder + f"/trial_{trial_number}.txt"

trial_file = create_trial_file()

with open(trial_file, "w") as f:
    f.write(f"Model architecture: {model}\n")
    f.write(f"Optimizer: {optimizer}\n")
    f.write(f"Criterion: {criterion}\n")
f.close()

# Accuracy calculation function
def calculate_metrics(y_pred, y_true):
    # Convert predictions to class indices (max value)
    predicted = torch.argmax(y_pred, dim=1)

    # Compute confusion matrix components
    tp = ((predicted == 1) & (y_true == 1)).sum().item()  # True Positive
    tn = ((predicted == 0) & (y_true == 0)).sum().item()  # True Negative
    fp = ((predicted == 1) & (y_true == 0)).sum().item()  # False Positive
    fn = ((predicted == 0) & (y_true == 1)).sum().item()  # False Negative

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return fp, tp, fn, tn, accuracy, precision, recall, specificity, f1

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    epoch_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'specificity': 0, 'f1': 0}

    for batch_X, batch_Y in train_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Calculate and aggregate metrics for the current batch
        fp, tp, fn, tn, accuracy, precision, recall, specificity, f1 = calculate_metrics(outputs, batch_Y)
        epoch_metrics['accuracy'] += accuracy
        epoch_metrics['precision'] += precision
        epoch_metrics['recall'] += recall
        epoch_metrics['specificity'] += specificity
        epoch_metrics['f1'] += f1

    # Average metrics over the dataset
    epoch_loss /= len(train_loader)
    epoch_metrics = {k: v / len(train_loader) for k, v in epoch_metrics.items()}

    # Log metrics using TensorBoard
    writer.add_scalar("Loss/epoch", epoch_loss, epoch)
    writer.add_scalar("Accuracy/epoch", epoch_metrics['accuracy'], epoch)
    writer.add_scalar("Precision/epoch", epoch_metrics['precision'], epoch)
    writer.add_scalar("Recall/epoch", epoch_metrics['recall'], epoch)
    writer.add_scalar("Specificity/epoch", epoch_metrics['specificity'], epoch)
    writer.add_scalar("F1/epoch", epoch_metrics['f1'], epoch)

    # Print metrics
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, '
          f'Accuracy: {epoch_metrics["accuracy"]*100:.2f}%, '
          f'Precision: {epoch_metrics["precision"]:.4f}, '
          f'Recall: {epoch_metrics["recall"]:.4f}, '
          f'Specificity: {epoch_metrics["specificity"]:.4f}, '
          f'F1 Score: {epoch_metrics["f1"]:.4f}')

    # Save metrics to trial file
    with open(trial_file, "a") as f:
        f.write(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, '
                f'Accuracy: {epoch_metrics["accuracy"]*100:.2f}%, '
                f'Precision: {epoch_metrics["precision"]:.4f}, '
                f'Recall: {epoch_metrics["recall"]:.4f}, '
                f'Specificity: {epoch_metrics["specificity"]:.4f}, '
                f'F1 Score: {epoch_metrics["f1"]:.4f}\n')

# Testing loop
model.eval()
test_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'specificity': 0, 'f1': 0}

with torch.no_grad():
    for batch_X, batch_Y in test_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        outputs = model(batch_X)

        # Calculate and aggregate metrics for the current batch
        fp, tp, fn, tn, accuracy, precision, recall, specificity, f1 = calculate_metrics(outputs, batch_Y)
        test_metrics['accuracy'] += accuracy
        test_metrics['precision'] += precision
        test_metrics['recall'] += recall
        test_metrics['specificity'] += specificity
        test_metrics['f1'] += f1

# Average metrics over the test set
test_metrics = {k: v / len(test_loader) for k, v in test_metrics.items()}

# Log metrics for testing
writer.add_scalar("Test Accuracy", test_metrics['accuracy'], num_epochs)
writer.add_scalar("Test Precision", test_metrics['precision'], num_epochs)
writer.add_scalar("Test Recall", test_metrics['recall'], num_epochs)
writer.add_scalar("Test Specificity", test_metrics['specificity'], num_epochs)
writer.add_scalar("Test F1", test_metrics['f1'], num_epochs)

# Print and save test metrics
print(f'Test Accuracy: {test_metrics["accuracy"]*100:.2f}%, '
      f'Test Precision: {test_metrics["precision"]:.4f}, '
      f'Test Recall: {test_metrics["recall"]:.4f}, '
      f'Test Specificity: {test_metrics["specificity"]:.4f}, '
      f'Test F1 Score: {test_metrics["f1"]:.4f}')

with open(trial_file, "a") as f:
    f.write(f'Test Accuracy: {test_metrics["accuracy"]*100:.2f}%, '
            f'Test Precision: {test_metrics["precision"]:.4f}, '
            f'Test Recall: {test_metrics["recall"]:.4f}, '
            f'Test Specificity: {test_metrics["specificity"]:.4f}, '
            f'Test F1 Score: {test_metrics["f1"]:.4f}\n')

writer.close()
