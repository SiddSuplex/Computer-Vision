# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 23:20:43 2020

@author: Siddhesh
"""

#Importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

import glob
import cv2

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, SpatialDropout2D
from keras.optimizers import Adam

import os
import random

#Specifying the data paths
train_img_dir = "images/train"
test_img_dir = "images/test"

#Setting the root directory and a data directory
root_dir = os.path.abspath('.')

#Creating a function ton preprocess images
def preprocess(image):
    
    #Reading the image
    img = cv2.imread(image)
    
    #Converting color image into a grayscale image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #Resizing all images to a size of 128 x 128
    img = cv2.resize(img, (32,32))
    
    #Performing histogram equalization on the image
    img = cv2.equalizeHist(img)
    
    #Converting images to float32
    img = img.astype('float32')
    
    #Normalizing the image
    img = img/255
    
    #Returning the image
    return img

#Creating a temporary array to store the images
temp = []

#Creating an empty array to store the labels
train_labels = []

#Initializing variable count
count = 1

#Iterating through the training images
for breed in glob.glob(train_img_dir + "/*"):
    for img in glob.glob(breed + "/*.jpg"):
        
        #Preprocessing the images
        img = preprocess(img)
        
        #Storing the preprocessed image in the temporary array
        temp.append(img)
        
        #Tracking the animals and the image count
        print(os.path.basename(breed), count)

        #Storing the labels in the empty array
        train_labels.append(os.path.basename(breed))

        #Incrementing count
        count += 1
        
#Stacking the images
train_img_arr = np.stack(temp)

#Creating a temporary array to store the images
temp = []

#Creating an empty array to store the labels
test_labels = []

#Initializing variable count
count = 1

#Iterating through the test images
for breed in glob.glob(test_img_dir + "/*"):
    for img in glob.glob(breed + "/*.jpg"):
        
        #Preprocessing the images
        img = preprocess(img)
        
        #Storing the preprocessed image in the temporary array
        temp.append(img)
        
        #Tracking the animals and image count
        print(os.path.basename(breed), count)

        #Storing the labels in the empty array
        test_labels.append(os.path.basename(breed))

        #Incrementing count
        count += 1
        
#Stacking the images
test_img_arr = np.stack(temp)

#Adding a depth of 1 in order to denote grayscale images
train_img_arr = np.reshape(train_img_arr, ((5544, 32, 32, 1)))
test_img_arr = np.reshape(test_img_arr, ((1840, 32, 32, 1)))

#Denoting the training and test data
train_data1 = train_img_arr
test_data1 = test_img_arr

#Converting the training and test labels into numpy array
train_labels1 = np.array(train_labels)
test_labels1 = np.array(test_labels)

#Confirming the shape of the images and labels
print(train_data1.shape)
print(train_labels1.shape)
print(test_data1.shape)
print(test_labels1.shape)

#Writing an error message
assert(train_data1.shape[0] == train_labels1.shape[0]), "Number of training images and labels don't match"
assert(test_data1.shape[0] == test_labels1.shape[0]), "Number of test images and labels don't match"

#Changing the labels to categorical values
lb = LabelEncoder()
train_labels1 = lb.fit_transform(train_labels1)
test_labels1 = lb.fit_transform(test_labels1)
train_labels1 = keras.utils.np_utils.to_categorical(train_labels1)
test_labels1 = keras.utils.np_utils.to_categorical(test_labels1)

#Defining parameters of the neural network
input_shape = train_data1.shape[1:4]
output_units = 37

#Creating a Neural Network
def Neural_Net(LR):
    
    #Creating a sequential model
    model = Sequential()

    #Adding a convolutional layer with 30 filters and a 5x5 kernel size with relu activation function
    model.add(Conv2D(240, (5,5), input_shape = input_shape , activation = 'elu'))
    #Adding a convolutional layer with 30 filters and a 5x5 kernel size with relu activation function
    model.add(Conv2D(240, (5,5), input_shape = input_shape , activation = 'elu'))
    #Adding a pooling layer with a 2x2 pool size
    model.add(MaxPooling2D(pool_size = (2,2)))
    #Adding a dropout layer
    model.add(Dropout(0.5))
    #Adding a convolutional layer with 15 filters and a 3x3 kernel size with relu activation function
    model.add(Conv2D(120, (3,3), activation = 'elu'))
    #Adding a convolutional layer with 15 filters and a 3x3 kernel size with relu activation function
    model.add(Conv2D(120, (3,3), activation = 'elu'))
    #Adding a pooling layer with a 2x2 pool size
    model.add(MaxPooling2D(pool_size = (2,2)))
    #Flattening the model
    model.add(Flatten())
    #Adding a fully connected dense layer with 300 nodes with relu activation function
    model.add(Dense(500, activation = 'elu'))
    #Adding a dropout layer
    model.add(Dropout(0.5))
    #Adding the output layer with 2 nodes with softmax activation function
    model.add(Dense(output_units, activation = 'softmax'))

    #Compiling the model
    model.compile(optimizer = Adam(lr = LR), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    #Returning the model
    return model

#Checking the summary of the model
model = Neural_Net(0.01)
print(model.summary())

#Creating a function to shuffle data
def shuffle(seed):
  random.seed(seed)
  random.shuffle(train_data1)
  random.seed(seed)
  random.shuffle(train_labels1)
  random.seed(seed)
  random.shuffle(test_data1)
  random.seed(seed)
  random.shuffle(test_labels1)

#Shuffling the data
shuffle(10)
shuffle(20)
shuffle(10)
shuffle(20)
shuffle(10)
shuffle(20)
shuffle(10)
shuffle(20)
shuffle(10)

#Fitting the data to the model
model.fit(train_data1, train_labels1, epochs = 3, validation_split = 0.2, shuffle = True)

#Calculating the testing accuracy
score = model.evaluate(test_data1, test_labels1, verbose = 0)
print("")
print("Test Accuracy is :", score[1])

#Got a test accuracy score of 0.9875 (98.75%)