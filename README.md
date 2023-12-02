# Road Damage Detection and Classification

This repository contains code for a road damage detection and classification model using TensorFlow and Keras. The project is divided into two main scripts: `main2.py` for training the model and `test2.py` for classifying road images.

## Features

- Road damage detection and classification using a Convolutional Neural Network (CNN).
- Utilizes TensorFlow and Keras for deep learning.
- Preprocessing functions for image data.
- Model training with visualizations of loss and accuracy.
- Classification script for predicting road conditions.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- tqdm
- scikit-learn
  
## Usage
- Training the Model (main2.py)
  
-Place your dataset in the appropriate folders (e.g., normal and damage folders).

-Configure the script by modifying the IMG_SIZE variable and other parameters if needed.

-The script will train the model, display loss and accuracy plots, and save the trained model as damageroad_detection_model.h5.

-Classifying Road Images (test2.py)

-Ensure you have a trained model (damageroad_detection_model.h5) in the same directory.

-Enter the path to the road image when prompted.

-The script will classify the road image and display the result.

## Additional Notes
The training script (main2.py) creates visualizations of randomly selected images with their corresponding labels for verification.
The classification script (test2.py) loads a pre-trained model and classifies a provided road image.
