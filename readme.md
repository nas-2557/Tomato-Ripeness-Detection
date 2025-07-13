\- N. Abinash, Partheban KV

# Tomato Ripeness Detection Using Transfer Learning

This project implements a tomato ripeness classification model using transfer learning. The objective is to classify tomatoes into different stages of ripeness based on images, supporting potential applications in agriculture, food quality control, and supply chain automation.

## Overview

The approach uses a pre-trained VGG16 model as a feature extractor. Features from the penultimate layer are passed into a Support Vector Machine (SVM) classifier, leveraging NVIDIA's cuML library for GPU-accelerated training. The model is trained and evaluated on a labeled dataset of tomato images, each annotated with its ripeness level.

## Features

- Transfer learning with VGG16 for robust feature extraction
- SVM classifier trained on top of deep features
- Dataset preprocessing with image augmentation
- Clean separation of training, validation, and test sets
- Support for GPU acceleration using cuML
- End-to-end training and evaluation pipeline

## Technologies

- Python
- TensorFlow / Keras
- cuML (RAPIDS.ai)
- scikit-learn
- NumPy, OpenCV
- Google Colab (for experimentation)

### Dataset: 
- Taken from kaggle (https://www.kaggle.com/datasets/enalis/tomatoes-dataset)
- Contains 7226 images of tomatoes in four different states: Unripe, Ripe, Old, and Damaged.
- For this model only images in the dataset within the subdirectories "Ripe" and "Unripe" are used for training the model.
- 90/5/5 Split (total: 3949)
	Ripe: 2195
	  Train: 1975
	  Test: 110
	  Validation: 110
	  
	Unripe: 1754
	 Train: 1585
	 Test: 84
	 Validation: 85

## Model Architecture

- **Feature extractor**: Pre-trained VGG16 with frozen convolutional layers
- **Classifier**: cuML SVM trained on flattened deep feature vectors

This hybrid design leverages the visual feature power of convolutional networks with the simplicity and speed of SVM classification.

## Results

The model achieved high classification accuracy on the held-out test set. Evaluation was conducted using standard metrics including accuracy and confusion matrix analysis.

- This model is based on this existing paper: https://www.ijisae.org/index.php/IJISAE/article/view/2538/1121


