# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 12:49:34 2019

@author: Siddhesh
"""

#Referred the tutorial offered at https://www.udemy.com/course/applied-deep-learningtm-the-complete-self-driving-car-course

#Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt

%tensorflow_version 1.x
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

import cv2
import pickle

from google.colab.patches import cv2_imshow

#Mounting google drive into google collaboratory
from google.colab import drive
drive.mount('/content/drive')

#Specifying the paths
train_path = "/content/drive/My Drive/Colab Notebooks/Computer Vision/Self Driving Car/german-traffic-signs/train.p"
validation_path = "/content/drive/My Drive/Colab Notebooks/Computer Vision/Self Driving Car/german-traffic-signs/valid.p"
test_path = "/content/drive/My Drive/Colab Notebooks/Computer Vision/Self Driving Car/german-traffic-signs/test.p"

#Unpickling the training data
with open(train_path, "rb") as f:
    train_data = pickle.load(f)
    
#Unpickling the validation data
with open(validation_path, "rb") as f:
    valid_data = pickle.load(f)
    
#Unpickling the testing data
with open(test_path, "rb") as f:
    test_data = pickle.load(f)
    
#Splitting the data into features and labels
x_train, y_train = train_data['features'], train_data['labels']
x_val, y_val = valid_data['features'], valid_data['labels']
x_test, y_test = test_data['features'], test_data['labels']

#Checking the shapes of the data
print(x_train.shape)
print(x_val.shape)
print(x_test.shape)
print("")
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)

#Checking the consistency of data
assert(x_train.shape[0] == y_train.shape[0]), "Dimensions of train data and labels differ"
assert(x_test.shape[0] == y_test.shape[0]), "Dimensions of test data and labels differ"
assert(x_train.shape[1:] == x_val.shape[1:] == x_test.shape[1:]), "Pixel size of train, validation and test don't match"

#Creating a function to preprocess the images
def preprocess(img):
    
    #Converting color image into a grayscale image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #Performing histogram equalization on the image
    img = cv2.equalizeHist(img)
    
    #Normalizing the image
    img = img/255
    
    #Returning the grayscale image
    return img

#Preprocessing the train, validation and test images
x_train = np.array(list(map(preprocess, x_train)))
x_val = np.array(list(map(preprocess, x_val)))
x_test = np.array(list(map(preprocess, x_test)))

#Reshaping the data to include the single channel of grayscale image
x_train = x_train.reshape(34799, 32, 32, 1)
x_val = x_val.reshape(4410, 32, 32, 1)
x_test = x_test.reshape(12630, 32, 32, 1)

#Converting are labels into categorical values
y_train = to_categorical(y_train, 43)
y_val = to_categorical(y_val, 43)
y_test = to_categorical(y_test, 43)

#Checking the shape of the data
print(x_train.shape)
print(x_val.shape)
print(x_test.shape)
print(y_train.shape)

#Setting parameters for the augmented images
datagen = ImageDataGenerator(width_shift_range = 0.1, height_shift_range = 0.1, zoom_range = 0.2, shear_range = 0.1, rotation_range = 10)

#Fitting image data to the data generator
datagen.fit(x_train)

#Creating the augmented images
augmented_images = datagen.flow(x_train, y_train, batch_size = 50)

#Creating a Neural Network
def Neural_Net(LR):
    #Creating a sequential model
    model = Sequential()

    #Adding a convolutional layer with 30 filters and a 5x5 kernel size with relu activation function
    model.add(Conv2D(240, (5,5), input_shape = (32, 32, 1) , activation = 'relu'))
    #Adding a convolutional layer with 30 filters and a 5x5 kernel size with relu activation function
    model.add(Conv2D(240, (5,5), input_shape = (32, 32, 1) , activation = 'relu'))
    #Adding a pooling layer with a 2x2 pool size
    model.add(MaxPooling2D(pool_size = (2,2)))
    #Adding a dropout layer
    model.add(Dropout(0.5))
    #Adding a convolutional layer with 15 filters and a 3x3 kernel size with relu activation function
    model.add(Conv2D(120, (3,3), activation = 'relu'))
    #Adding a convolutional layer with 15 filters and a 3x3 kernel size with relu activation function
    model.add(Conv2D(120, (3,3), activation = 'relu'))
    #Adding a pooling layer with a 2x2 pool size
    model.add(MaxPooling2D(pool_size = (2,2)))
    #Flattening the model
    model.add(Flatten())
    #Adding a fully connected dense layer with 300 nodes with relu activation function
    model.add(Dense(500, activation = 'relu'))
    #Adding a dropout layer
    model.add(Dropout(0.5))
    #Adding the output layer with 43 nodes with softmax activation function
    model.add(Dense(43, activation = 'softmax'))

    #Compiling the model
    model.compile(optimizer = Adam(lr = LR), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    #Returning the model
    return model

#Checking the summary of the model
model = Neural_Net(0.001)
print(model.summary())

#Fitting the training and validation data to the model
model.fit_generator(augmented_images, steps_per_epoch = 2000, epochs = 15, validation_data = (x_val , y_val), shuffle = 1)

#Calculating the testing accuracy
score = model.evaluate(x_test, y_test, verbose = 0)
print("")
print("Test Accuracy is :", score[1])

#Got a test accuracy score of 0.9832145685349205