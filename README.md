# Face Detection and Mask Classification Project

## Project Overview
This project leverages advanced computer vision techniques to detect human faces in images and classify them based on mask presence. It incorporates image preprocessing, face detection, dataset preparation, and the development of Convolutional Neural Network (CNN) models for mask classification, followed by comprehensive model training and evaluation.

## Table of Contents
- [Project Overview](#project-overview)
- [Components](#components)
- [CNN Models](#cnn-models)
- [Conclusion](#conclusion)




## Prerequisites

- Python 3.9
- OpenCV: `pip install opencv-python`
- TensorFlow or PyTorch: Install the appropriate deep learning framework based on the chosen model.

## Components

### Image Preprocessing
- **Techniques Used:** Resizing and cropping.
- **Tools/Libraries:** OpenCV.

### Face Detection
- **Model Used:** open source model that was Built using dlib's state-of-the-art face recognition built with deep learning.
- **Implementation Details:** The model was used to get the cropped faces out of the provided images.

### Dataset Preparation
- **Data Sources:** The images folder provided with the assignment
- **Preprocessing Steps:** Image augmentation and image cropping.

## CNN Models

## Model Architecture

### Initial Layers:
- **Conv2D Layers:** The initial layers are Conv2D layers, used for convolutional operations to extract features from the input images. The first Conv2D layer has 32 filters with a kernel size of (3,3) and uses the ReLU activation function. The input shape is tailored to the size of the processed images.
- **MaxPooling2D Layers:** Each Conv2D layer is followed by a MaxPooling2D layer with a pool size of (2,2). These layers reduce the spatial dimensions of the output, helping decrease the number of parameters and computational load.

### Deepening the Network:
- **Additional Conv2D and MaxPooling2D Layers:** The network depth is increased with more Conv2D and MaxPooling2D layers. The subsequent Conv2D layers have a higher number of filters (64), enabling the model to learn more complex features.
- **Flatten Layer:** After the convolutional and pooling layers, a Flatten layer is included to convert the 2D matrix data into a 1D vector, essential for the fully connected layers.

### Fully Connected Layers (ANN part):
- **Dense Layer with ReLU Activation:** The flattened output is fed into a Dense layer with 128 neurons, using the ReLU activation function. This layer is crucial for learning high-level features.
- **Dropout Layer:** A Dropout layer with a rate of 0.5 is added to prevent overfitting by randomly setting a fraction of input units to 0 during training.
- **Output Dense Layer:** The final layer is a Dense layer with a single neuron, using the sigmoid activation function, appropriate for binary classification tasks ('mask' or 'no mask').

## Compilation and Optimization
- **Loss Function:** The model employs binary_crossentropy for its loss function, standard for binary classification.
- **Optimizer:** The Adam optimizer is used for its effectiveness in managing sparse gradients and adapting learning rates.
- **Metrics:** Model performance is evaluated using accuracy as the primary metric.

## Training
- The model is trained on a dataset categorized into 'mask' and 'no mask'.
- Specific training details such as batch sizes, number of epochs, and strategies are crucial for a full understanding but are not included in the provided snippets.

## Evaluation
- The model's effectiveness is assessed using accuracy and other relevant metrics, determining its capability to accurately classify images into 'mask' and 'no mask' categories.

## Conclusion
- **Summary of Findings:** The models were capable of extracting the features correctly even though it was concerning since the images after cropping seemed to be not so clear, the classification gave very good results when it comes to the different metrics.
- **Applications:** This project can be used for surveillance in times of pandemics such as the time of COVID and it could be very helpful in the cities that lack clean air and masks are a must for protecting people and children.


