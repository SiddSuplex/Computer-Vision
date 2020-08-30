#!/usr/bin/env python
# coding: utf-8

# ### Data Preprocessing

# In[1]:


#Importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import glob
from keras.utils import to_categorical


# In[2]:


#Changing the directory
os.chdir("C:/Users/Siddhesh/Desktop/Siddhesh/Projects/Face Mask Detection/Project/Data")


# In[3]:


#Specifying the file paths
train_path = "train"
val_path = "val"
test_path = "test"


# In[4]:


#Creating a function to preprocess images
def preprocess(img_path):
    
    #Reading the image
    img = cv2.imread(img_path)
    
    #Resizing the image
    img = cv2.resize(img, (100, 100))
    
    #Converting the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #Normalizing the image
    gray_img = gray_img/255
    
    #Returning the normalized image
    return gray_img


# In[5]:


#Creating a function to retrieve data
def get_data(path):
    
    #Creating empty arrays to store the data and labels    
    data = []
    labels = []
    
    #Trying the operation
    try:

        #Iterating through the images
        for i in os.listdir(path):
            for j in glob.glob(path + "/" + i):
                for k in glob.glob(j + "/*.jpg"):

                    #Adding the images and labels to the respective arrays
                    data.append(preprocess(k))
                    labels.append(i)
                    
    #Handling the exception
    except Exception as e:
        
        print("Unable to preprocess current image!", e)
        
    #Returning the data and labels
    return np.array(data), np.array(labels)

#Creating a function to reshape data
def reshape_data(data):
    
    #Retrieving the shape of the data
    shape = data.shape
    
    #Reshaping the data
    data = data.reshape((shape[0], shape[1], shape[2], 1))
    
    #Returning the data
    return data

#Creating a function to map labels to values
def label_map(labels):

    #Creating an empty array
    arr = np.zeros(shape = (len(labels), ))

    #Finding the unique labels
    unique_labels = np.unique(labels)

    #Iterating through the unique labels
    for u_index in range(0, len(unique_labels)):

        #Iterating through the labels
        for l_index in range(0, len(labels)):

            #Checking whether label is alien
            if(labels[l_index] == unique_labels[u_index]):
                arr[l_index] = u_index

    #Returning the new labels
    return arr

#Creating a function to perform one-hot-encoding
def encode(labels, num_classes):
    
    #Mapping the labels to numbers
    labels = label_map(labels)
    
    #Encoding the labels
    encoded_labels = to_categorical(labels, num_classes = num_classes)
    
    #Returning the encoded labels
    return encoded_labels


# In[6]:


#Retrieving the data
train_data, train_labels = get_data(path = train_path)
val_data, val_labels = get_data(path = val_path)
test_data, test_labels = get_data(path = test_path)


# In[7]:


#Reshaping the data
train_data = reshape_data(train_data)
val_data = reshape_data(val_data)
test_data = reshape_data(test_data)

#Performing one hot encoding on labels
train_labels = encode(train_labels, num_classes = 2)
val_labels = encode(val_labels, num_classes = 2)
test_labels = encode(test_labels, num_classes = 2)


# In[8]:


#Saving the data
np.save("train_data", train_data)
np.save("validation_data", val_data)
np.save("test_data", test_data)

#Saving the labels
np.save("train_labels", train_labels)
np.save("validation_labels", val_labels)
np.save("test_labels", test_labels)

