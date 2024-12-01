import os
import torch
import numpy as np
import pickle
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

# Set the root directory for the dataset
root_dir = 'archive'

# Image preprocessing transform (resizing to a standard size and normalizing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale 
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize pixel values to [0, 1]
])

# Initialize lists to store images and labels
images = []
labels = []

# Define class names and label mapping
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
label_map = {name: i for i, name in enumerate(class_names)}

# Loop over the subdirectories in 'Training' and 'Testing' folders
for split in ['Training', 'Testing']:
    for class_name in class_names:
        class_dir = os.path.join(root_dir, split, class_name)
        label = label_map[class_name]
        
        # Loop through all images in the class directory
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                # Open and preprocess the image
                img = Image.open(img_path).convert('RGB')
                img = transform(img)
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
        
# Convert lists to tensors
X = torch.stack(images)
y = torch.tensor(labels, dtype=torch.long)

# Manually split the data into train and test sets (80% train, 20% test)
# Shuffle the indices for random splitting
num_samples = len(X)
indices = torch.randperm(num_samples).tolist()

# 80% for training and 20% for testing
split_index = int(0.8 * num_samples)
train_indices = indices[:split_index]
test_indices = indices[split_index:]

# Use the indices to split the data
X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]

# Save the data as pickle files
pickle_folder = 'pickle/'

# Ensure the folder exists
os.makedirs(pickle_folder, exist_ok=True)

# Save training and testing data
with open(os.path.join(pickle_folder, 'X_train.pickle'), 'wb') as f:
    pickle.dump(X_train.numpy(), f)

with open(os.path.join(pickle_folder, 'Y_train.pickle'), 'wb') as f:
    pickle.dump(y_train.numpy(), f)

with open(os.path.join(pickle_folder, 'X_test.pickle'), 'wb') as f:
    pickle.dump(X_test.numpy(), f)

with open(os.path.join(pickle_folder, 'Y_test.pickle'), 'wb') as f:
    pickle.dump(y_test.numpy(), f)

print("Pickle files have been created successfully!")
