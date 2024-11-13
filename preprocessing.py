import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
import pickle
import re
import shutil
import pickle

os.chdir("CSCI167Project")
print("\n\n", os.getcwd())
print(os.listdir())

X_train = []
Y_train = []
X_test = []
Y_test = []

img_folder = os.path.join(os.getcwd(), "images")
train_folder = os.path.join(img_folder, "Training")
test_folder = os.path.join(img_folder, "Testing")

def getImages(train_or_test):
    # NEW_HEIGHT =  
    # NEW_WIDTH = 
    for tumor_dir in os.listdir(train_or_test):
        tumor_dir_path = os.path.join(train_or_test, tumor_dir)
        for img in os.listdir(tumor_dir_path):
            img_path = os.path.join(tumor_dir_path, img)
    
print(getImages(train_folder))
