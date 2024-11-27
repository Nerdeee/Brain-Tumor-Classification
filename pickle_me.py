import os
import pickle
import numpy as np
import cv2  # Use OpenCV to handle image resizing if needed

# Define the paths
data_dir = "archive/"
training_dir = os.path.join(data_dir, "Training")
testing_dir = os.path.join(data_dir, "Testing")
classes_to_include = ["meningioma", "notumor"]  # Relevant classes
label_map = {"meningioma": 1, "notumor": 0}  # Binary labels

def load_images_from_directory(directory, classes_to_include, label_map, target_size=(128, 128)):
    """
    Loads images and their corresponding labels from a directory.
    Args:
        directory: Directory containing class subfolders.
        classes_to_include: List of class subfolders to include.
        label_map: Mapping of class names to labels.
        target_size: Target size for image resizing.
    Returns:
        (images, labels): Tuple of lists containing image data and corresponding labels.
    """
    images = []
    labels = []
    
    for class_name in classes_to_include:
        class_dir = os.path.join(directory, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
            img = cv2.resize(img, target_size)  # Resize to consistent size
            images.append(img)
            labels.append(label_map[class_name])
    
    return np.array(images), np.array(labels)

# Load and process training data
X_train, Y_train = load_images_from_directory(training_dir, classes_to_include, label_map)

# Load and process testing data
X_test, Y_test = load_images_from_directory(testing_dir, classes_to_include, label_map)

# Normalize pixel values to [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Save processed data to pickle files
pickle_folder = "meningioma-pickle/"
os.makedirs(pickle_folder, exist_ok=True)

with open(os.path.join(pickle_folder, "X_train.pickle"), "wb") as f:
    pickle.dump(X_train, f)

with open(os.path.join(pickle_folder, "Y_train.pickle"), "wb") as f:
    pickle.dump(Y_train, f)

with open(os.path.join(pickle_folder, "X_test.pickle"), "wb") as f:
    pickle.dump(X_test, f)

with open(os.path.join(pickle_folder, "Y_test.pickle"), "wb") as f:
    pickle.dump(Y_test, f)

print("Pickle files for meningioma and notumor saved successfully!")
