├── dataset/
│   ├── train/
│   │   ├── cat/
│   │   └── dog/
├── train.py       # Main script for training and testing the model
├── split.py       # Script for splitting the dataset into train/test sets
└── README.md      # This file

Dataset
The dataset consists of images of cats and dogs, which have been divided into two folders: cat and dog. Before training, we preprocess the dataset to resize the images to 128x128 pixels, convert them to tensors, and normalize them using a mean and standard deviation of 0.5 for each channel.

To split the dataset into an 8:2 ratio for training and testing, use the split.py script.

Prerequisites
To run this project, you will need to have the following installed:
Python 3.11
PyTorch
Torchvision
Numpy

Model Architecture
The CNN used for classification consists of the following layers:
Convolution Layer 1: 3 input channels, 32 output channels, 3x3 kernel size
Max Pooling Layer 1: 2x2 pooling
Convolution Layer 2: 32 input channels, 64 output channels, 3x3 kernel size
Max Pooling Layer 2: 2x2 pooling
Fully Connected Layer 1: 64 * 32 * 32 input features, 512 output features
Fully Connected Layer 2: 512 input features, 2 output features (for cat and dog classes)

Training
To train the model, run:
python train.py

This project implements a simple convolutional neural network (CNN) using PyTorch to classify images of cats and dogs. The model is trained on a dataset split into training and testing sets with an 8:2 ratio. The project includes basic preprocessing, model definition, training, and testing.




