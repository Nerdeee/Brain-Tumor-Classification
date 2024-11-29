import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
import pickle
import re
from torchvision import transforms

print('\n Current directory: ', os.listdir())

def create_pickle(X_train, Y_train, X_test, Y_test):
    pickle_dir = os.path.join(os.getcwd(), "pickle")
    os.makedirs(pickle_dir, exist_ok=True)

    X_train_path = os.path.join(pickle_dir, "X_train.pickle")
    Y_train_path = os.path.join(pickle_dir, "Y_train.pickle")
    X_test_path = os.path.join(pickle_dir, "X_test.pickle")
    Y_test_path = os.path.join(pickle_dir, "Y_test.pickle")

    with open(X_train_path, "wb") as f:
        pickle.dump(X_train, f)

    with open(Y_train_path, "wb") as f:
        pickle.dump(Y_train, f)

    with open(X_test_path, "wb") as f:
        pickle.dump(X_test, f)

    with open(Y_test_path, "wb") as f:
        pickle.dump(Y_test, f)
    
def oneHotEncode(tumor_type):
    encoding = [0] * 4

    encoding_dictionary = {
        "glioma": 0,
        "meningioma": 1,
        "notumor": 2,
        "pituitary": 3
    }

    encoding[encoding_dictionary[tumor_type]] = 1
    return encoding

def findMinImageCount(train_or_test):
    image_counts = {}
    for tumor_dir in os.listdir(train_or_test):
        tumor_dir_path = os.path.join(train_or_test, tumor_dir)
        if os.path.isdir(tumor_dir_path):
            num_images = len(os.listdir(tumor_dir_path))
            image_counts[tumor_dir] = num_images
    min_images = min(image_counts.values())
    print("Image counts per directory:", image_counts)
    print("Minimum number of images:", min_images)
    return min_images

def getImages(train_or_test, X_features, Y_labels):
    NEW_HEIGHT = 168    # fixed height
    NEW_WIDTH = 150     # fixed width
    CROP_PIXELS = 20
    CROP_BOTTOM = 50
    images_loaded = 0
    for tumor_dir in os.listdir(train_or_test):
        tumor_dir_path = os.path.join(train_or_test, tumor_dir)
        num_images_for_folder = 0
        if os.path.isdir(tumor_dir_path):
            for img in os.listdir(tumor_dir_path):
                img_path = os.path.join(tumor_dir_path, img)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    """ height, width, _ = image.shape
                    # Update the minimum dimensions
                    if height < min_height:
                        min_height = height
                    if width < min_width:
                        min_width = width """
                    image = cv2.resize(image, (NEW_WIDTH, NEW_HEIGHT))
                    image = image[CROP_PIXELS:-CROP_BOTTOM, CROP_PIXELS:-CROP_PIXELS]
                    image = image / 255
                    oneHotArray = oneHotEncode(tumor_dir)
                    oneHotArray = np.array(oneHotArray)
                    X_features.append(image)
                    Y_labels.append(oneHotArray)
                    images_loaded += 1
                    num_images_for_folder += 1
        print(f'Total images in directory {tumor_dir}: ', num_images_for_folder)
    print(f'Total images for {train_or_test}: ', images_loaded)
                    

X_train = []
Y_train = []
X_test = []
Y_test = []

img_folder = os.path.join(os.getcwd(), "images")
train_folder = os.path.join(img_folder, "Training")
test_folder = os.path.join(img_folder, "Testing")

min_images_train = findMinImageCount(train_folder)
min_images_test = findMinImageCount(test_folder)

getImages(train_folder, X_train, Y_train)
getImages(test_folder, X_test, Y_test)

print(len(X_train))
print(len(Y_train))
print(len(X_test))
print(len(Y_test))

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

create_pickle(X_train, Y_train, X_test, Y_test)