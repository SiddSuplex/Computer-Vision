#!/usr/bin/env python
# coding: utf-8

# ### Neural Network

# In[1]:


#Importing required libraries
import numpy as np
import os
#%tensorflow_version 1.x
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


# In[2]:


#Mounting google drive into google collaboratory
from google.colab import drive
drive.mount('/content/drive')


# In[3]:


#Changing the directory
os.chdir("/content/drive/My Drive/Colab Notebooks/Computer Vision/Projects/Face Mask Detection/Data")


# In[4]:


#Loading the data
train_data = np.load("train_data.npy")
val_data = np.load("validation_data.npy")
test_data = np.load("test_data.npy")

#Loading the labels
train_labels = np.load("train_labels.npy")
val_labels = np.load("validation_labels.npy")
test_labels = np.load("test_labels.npy")


# In[5]:


#Creating a Neural Network
def Neural_Net(lr, data = train_data):

    #Creating a sequential model
    model = Sequential()

    #Adding a convolutional layer with 240 filters and a 5x5 kernel size with relu activation function
    model.add(Conv2D(filters = 240, kernel_size = (5,5), input_shape = train_data.shape[1:] , activation = 'relu'))

    #Adding a convolutional layer with 240 filters and a 5x5 kernel size with relu activation function
    model.add(Conv2D(filters = 240, kernel_size = (5,5), activation = 'relu'))
    
    #Adding a pooling layer with a 2x2 pool size
    model.add(MaxPooling2D(pool_size = (2,2)))
    
    #Adding a dropout layer
    model.add(Dropout(0.5))
    
    #Adding a convolutional layer with 120 filters and a 3x3 kernel size with relu activation function
    model.add(Conv2D(filters = 120, kernel_size = (3,3), activation = 'relu'))
    
    #Adding a convolutional layer with 120 filters and a 3x3 kernel size with relu activation function
    model.add(Conv2D(filters = 120, kernel_size = (3,3), activation = 'relu'))
    
    #Adding a pooling layer with a 2x2 pool size
    model.add(MaxPooling2D(pool_size = (2,2)))
    
    #Flattening the model
    model.add(Flatten())
    
    #Adding a fully connected dense layer with 500 nodes with relu activation function
    model.add(Dense(500, activation = 'relu'))
    
    #Adding a dropout layer
    model.add(Dropout(0.5))
    
    #Adding the output layer with 2 nodes with softmax activation function
    model.add(Dense(2, activation = 'softmax'))
    
    #Compiling the model
    model.compile(optimizer = Adam(lr = lr), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    #Returning the model
    return model

#Creating the model
model = Neural_Net(lr = 0.001)

#Displaying the summary of the model
print(model.summary())


# In[6]:


#Creating a checkpoint to save the lowest validation loss
checkpoint = ModelCheckpoint('model-{epoch:03d}.model', monitor = 'val_loss', verbose = 0, save_best_only = True, mode = 'auto')

#Fitting the training and validation data to the model
model.fit(x = train_data, y = train_labels, epochs = 10, callbacks = [checkpoint], validation_data = (val_data, val_labels))


# In[7]:


#Evaluating the model
score = model.evaluate(test_data, test_labels)

#Printing the accuracy of the model
print(f"Accuracy of the model is {score[1] * 100}%")


# In[8]:


#Saving the model
model.save("neural_net_model.h5")

