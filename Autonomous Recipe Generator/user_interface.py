# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:33:30 2020

@author: Siddhesh
"""

#Importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

import glob
import cv2

#%tensorflow_version 1.x
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, SpatialDropout2D
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from keras.utils import to_categorical

import os
import random

import pickle

#Referred https://machinelearningmastery.com/save-load-keras-deep-learning-models/

#Loading the data
food_labels = np.array(pd.read_csv("labels.txt", header = None))

#Loading the model from disk
#vgg_model = load_model('vgg_model.h5')
extra_trees_model = pickle.load(open('extra_trees_model.sav', 'rb'))

#Loading the sample image path
img_path = "samosa.jpg"

#Creating a function to preprocess images
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

#Preprocessing the image
img = preprocess(img_path)

#Defining the input shape
input_shape = img.shape

#Defining the output units
output_units = 101

#Creating a Neural Network
def VGG_Net(LR):
    
    #Creating a sequential model
    model = Sequential()

    #Creating a vgg model
    #vgg_model = VGG16(weights = 'vgg_model.h5', include_top = False)
    
    vgg_model = VGG16(input_shape = input_shape, include_top = False)
    
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
vgg_model = VGG_Net(0.01)
print(vgg_model.summary())

#Loading the pre-trained weights
vgg_model.load_weights('vgg_model.h5')

#Compiling the model
vgg_model.compile(optimizer = Adam(lr = 0.01), loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Reshaping the image
img = img.reshape(1,32,32,3)

#Defining the label encoder
lb = LabelEncoder()
food_labels = lb.fit_transform(food_labels)

#Predicting the food product
pred1 = vgg_model.predict_classes(img)

#Retrieving the name of the food
food_name = lb.inverse_transform(pred1)
print(food_name)

print("\n")

#Predicting the food type
pred2 = extra_trees_model.predict([pred1])

#Retrieving the type of the food
print(pred2)
