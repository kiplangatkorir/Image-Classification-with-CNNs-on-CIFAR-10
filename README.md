# Image Classification with CNNs on CIFAR-10

This project demonstrates how to train a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset using TensorFlow/Keras.

## Project Overview

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 test images. This project involves:
- Loading and preprocessing the CIFAR-10 dataset
- Building a CNN model using TensorFlow/Keras
- Training the model and evaluating its performance
- Visualizing training history and prediction results

## Getting Started

### Prerequisites

To run this project, you need the following libraries:
- TensorFlow
- NumPy
- Matplotlib

You can install these using pip:

```bash
pip install tensorflow numpy matplotlib

```
### Running the Project

1. Clone the repository:

```bash
git clone https://github.com/yourusername/cifar10-cnn.git
cd cifar10-cnn
```

2. Open the project in Google Colab and run the code cells in 'cifar10_cnn.ipynb' to train and evaluate the model.

### Project Structure

```bash

├── cifar10_cnn.ipynb   # Jupyter Notebook with the complete project code
├── README.md           # Project description and instructions

```
### Model Architecture

The CNN model consists of the following layers:

-Convolutional layer with 32 filters, kernel size of 3x3, and ReLU activation

-MaxPooling layer with pool size of 2x2

-Convolutional layer with 64 filters, kernel size of 3x3, and ReLU activation

-MaxPooling layer with pool size of 2x2

-Convolutional layer with 128 filters, kernel size of 3x3, and ReLU activation

-Flatten layer

-Dense layer with 128 units and ReLU activation

-Dropout layer with a rate of 0.5

-Dense output layer with 10 units and softmax activation

## Results
The model achieves an accuracy of approximately 70% on the test dataset after 10 epochs.

## Visualizations
Training and validation accuracy and loss curves, as well as sample predictions, are plotted to provide insights into the model's performance.

## License
This project is licensed under the MIT License - see the LICENSE file for details.


