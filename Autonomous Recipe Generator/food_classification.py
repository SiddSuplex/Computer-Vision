# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:22:46 2020

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
from keras.applications.vgg16 import VGG16

import os
import random

#Specifying the data paths
train_img_dir = "/kaggle/input/food41/images"

#Setting the root directory and a data directory
root_dir = os.path.abspath('.')

#Creating a function ton preprocess images
def preprocess(image):
    
    #Reading the image
    img = cv2.imread(image)
    
    #Resizing all images to a size of 128 x 128
    img = cv2.resize(img, (32,32))
    
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
for food in glob.glob(train_img_dir + "/*"):
    for img in glob.glob(food + "/*.jpg"):
        
        print(os.path.basename(img))

        #Preprocessing the images
        img = preprocess(img)
        
        #Storing the preprocessed image in the temporary array
        temp.append(img)
        
        #Tracking the animals and the image count
        print(os.path.basename(food), count, )

        #Storing the labels in the empty array
        train_labels.append(os.path.basename(food))

        #Incrementing count
        count += 1
        
#Stacking the images
train_img_arr = np.stack(temp)

#Adding a depth of 1 in order to denote grayscale images
train_img_arr = np.reshape(train_img_arr, ((101000, 32, 32, 3)))

#Denoting the training and test data
train_data1 = train_img_arr

#Converting the training and test labels into numpy array
train_labels1 = np.array(train_labels)

#Confirming the shape of the images and labels
print(train_data1.shape)
print(train_labels1.shape)

#Writing an error message
assert(train_data1.shape[0] == train_labels1.shape[0]), "Number of training images and labels don't match"

#Changing the labels to categorical values
lb = LabelEncoder()
train_labels1 = lb.fit_transform(train_labels1)
train_labels1 = keras.utils.np_utils.to_categorical(train_labels1)

#Defining parameters of the neural network
input_shape = train_data1.shape[1:4]
output_units = 101

#Creating a Neural Network
def VGG_Net(LR):
    
    #Creating a sequential model
    model = Sequential()

    #Creating a vgg model
    vgg_model = VGG16(input_shape = input_shape, weights = 'imagenet', include_top = False)
    
    #Iterating through the vgg model
    for layers in vgg_model.layers:
        model.add(layers)
        
    #Flattening the model
    model.add(Flatten())
    
    #Adding the output layer with 101 nodes with softmax activation function
    model.add(Dense(output_units, activation = 'softmax'))

    #Compiling the model
    model.compile(optimizer = Adam(lr = LR), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    #Returning the model
    return model

#Checking the summary of the model
model = VGG_Net(0.01)
print(model.summary())

#Creating a function to shuffle data
def shuffle(seed):
  random.seed(seed)
  random.shuffle(train_data1)
  random.seed(seed)
  random.shuffle(train_labels1)
  
#Shuffling the data
shuffle(10)
shuffle(20)
shuffle(10)
shuffle(20)
shuffle(10)
shuffle(20)
shuffle(10)
shuffle(20)

#Fitting the data to the model
model.fit(train_data1, train_labels1, epochs = 3, validation_split = 0.2, shuffle = True)

#Got a training accuracy score of 0.9650 (96.5%) and validation accuracy score of 0.9169 (91.69%)

#Needs Improvement