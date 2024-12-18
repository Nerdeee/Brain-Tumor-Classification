# Brain Tumor Detection Using CNNs

This project uses a Convolutional Neural Network (CNN) to classify brain MRI images into four categories: **glioma**, **meningioma**, **pituitary tumor**, and **no tumor**. The goal is to leverage deep learning for the early and accurate detection of brain tumors, aiding medical professionals in diagnosis and treatment planning.

---

## Table of Contents
- [Brain Tumor Detection Using CNNs](#brain-tumor-detection-using-cnns)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Dataset](#dataset)
    - [Key Details:](#key-details)
  - [Model 1 Architecture](#model-1-architecture)
    - [Hyperparameters](#hyperparameters)
    - [Results](#results)
  - [Model 2 Architecture](#model-2-architecture)
    - [Hyperparameters](#hyperparameters-1)
    - [Results](#results-1)
  - [Installation](#installation)

---

## Overview
Brain tumors can be detected using MRI scans. By training a CNN on a labeled dataset of MRI images, this project aims to automate and enhance the accuracy of tumor classification. Two CNN models with varying architectures were evaluated and compared to determine the best-performing approach.

---

## Dataset
The dataset used is the **Brain Tumor MRI Dataset**, which combines images from three sources: Figshare, SARTAJ, and Br35H datasets. It contains **7,023 grayscale MRI images** distributed across four classes:
- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

<img src="dataset-preview/dataset-preview.png"/>

### Key Details:
- MRI images are RGB.
- Images were resized for uniformity during preprocessing.
- Challenges included the different profiles of images.

---

## Model 1 Architecture
The chosen CNN model architecture includes:
1. **Convolutional Layers**: 4 convolutional layers with a kernel of size 2x2 with a stride of 2 and padding of 1.
2. **Pooling**: Max pooling with a kernel size of 2 to reduce feature map dimensions and computational cost.
3. **Fully Connected Layers**: One dense layers for classification into one of four classes.
4. **Activation Functions**: ReLU is used for non-linearity, and the final layer uses Softmax for multi-class classification.

---

### Hyperparameters
- **Batch Size**: 96
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss

<img src="models/diagrams/model_1.png" width=65%>

---

### Results
Key metrics used to evaluate model performance:
- **Accuracy**: Over 99% on the test set after resolving data preprocessing issues.
- **Recall**: Less than optimal performance near 50%.

<img src="models/metric-graphs/model1/test-accuracy.png">
<img src="models/metric-graphs/model1/test-recall.png">

---

## Model 2 Architecture
The chosen CNN model architecture includes:
1. **Convolutional Layers**: 5 convolutional layers using kernels of size 3x3 with a stride of 1 and padding of 1.
2. **Pooling**: Max pooling with a kernel size of 3.
3. **Fully Connected Layers**: Two dense layers for classification into one of four classes.
4. **Activation Functions**: ReLU is used for non-linearity, and the final layer uses Softmax for multi-class classification.

---

### Hyperparameters
- **Batch Size**: 96
- **Learning Rate**: 0.0001
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss

<img src="models/diagrams/model_2.png" width=65%>

---

### Results
Key metrics used to evaluate model performance:
- **Accuracy**: Over 99% on the test set after resolving data preprocessing issues.
- **Recall**: Less than optimal performance near 50%.

<img src="models/metric-graphs/model2/test-accuracy.png">
<img src="models/metric-graphs/model2/test-recall.png">


---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Nerdeee/Brain-Tumor-Classification.git
   cd brain-tumor-detection
