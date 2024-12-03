import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

if torch.cuda.is_available():
    print(f"\nGPU: {torch.cuda.get_device_name(0)} is available")
    device = torch.device("cuda")
else:
    print('\nNo GPU available')
    device = torch.device("cpu")

writer = SummaryWriter()

pickle_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pickle')

# Load pickled datasets
with open(pickle_folder + "/X_train.pickle", "rb") as f:
    X_train = pickle.load(f)
with open(pickle_folder + "/Y_train.pickle", "rb") as f:
    Y_train = pickle.load(f)
with open(pickle_folder + "/X_test.pickle", "rb") as f:
    X_test = pickle.load(f)
with open(pickle_folder + "/Y_test.pickle", "rb") as f:
    Y_test = pickle.load(f)

# Prepare data as tensors
X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()

# Check if the data already has a channel dimension (e.g., (N, H, W))
if X_train.ndimension() == 3:  # (N, H, W)
    X_train = X_train.unsqueeze(1)  # Add channel dimension (N, 1, H, W)
if X_test.ndimension() == 3:  # (N, H, W)
    X_test = X_test.unsqueeze(1)  # Add channel dimension (N, 1, H, W)

# Ensure long type for classification
Y_train = torch.tensor(Y_train, dtype=torch.long)
Y_test = torch.tensor(Y_test, dtype=torch.long)


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
        x = self.dropout(x)  # Dropout
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
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Log model details
print(model)
print(optimizer)
print(criterion)
print(f"Batch Size: {bs}")

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
    train_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'specificity': 0, 'f1': 0}
    test_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'specificity': 0, 'f1': 0}
    
    # Training phase
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_Y in train_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # Calculate metrics for the current batch
        fp, tp, fn, tn, accuracy, precision, recall, specificity, f1 = calculate_metrics(outputs, batch_Y)
        train_metrics['accuracy'] += accuracy
        train_metrics['precision'] += precision
        train_metrics['recall'] += recall
        train_metrics['specificity'] += specificity
        train_metrics['f1'] += f1

    # Average metrics over the training set
    epoch_loss /= len(train_loader)
    train_metrics = {k: v / len(train_loader) for k, v in train_metrics.items()}

    # Testing phase
    model.eval()
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            outputs = model(batch_X)

            # Calculate metrics for the current batch
            fp, tp, fn, tn, accuracy, precision, recall, specificity, f1 = calculate_metrics(outputs, batch_Y)
            test_metrics['accuracy'] += accuracy
            test_metrics['precision'] += precision
            test_metrics['recall'] += recall
            test_metrics['specificity'] += specificity
            test_metrics['f1'] += f1

    # Average metrics over the test set
    test_metrics = {k: v / len(test_loader) for k, v in test_metrics.items()}

    # Log metrics to TensorBoard for training
    writer.add_scalar("Train/Loss", epoch_loss, epoch)
    for metric_name, metric_value in train_metrics.items():
        writer.add_scalar(f"Train/{metric_name.capitalize()}", metric_value, epoch)

    # Log metrics to TensorBoard for testing
    for metric_name, metric_value in test_metrics.items():
        writer.add_scalar(f"Test/{metric_name.capitalize()}", metric_value, epoch)

    # Print epoch results
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Loss: {epoch_loss:.4f}, '
          f'Train Accuracy: {train_metrics["accuracy"]*100:.2f}%, '
          f'Test Accuracy: {test_metrics["accuracy"]*100:.2f}%, '
          f'Train Recall: {train_metrics["recall"]:.4f}, '
          f'Test Recall: {test_metrics["recall"]:.4f}, '
          f'Train F1 Score: {train_metrics["f1"]:.4f}, '
          f'Test F1 Score: {test_metrics["f1"]:.4f}')
    
writer.close()
