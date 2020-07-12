# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:05:45 2020

@author: Siddhesh
"""

#Referred https://www.udemy.com/course/advanced-computer-vision/learn

#Importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

import glob
import cv2

%tensorflow_version 1.x
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, SpatialDropout2D, LeakyReLU, BatchNormalization
from keras.optimizers import Adam, SGD

import os
import random

#Loading the mnist data
mnist = keras.datasets.mnist

#Separating the mnist data into train and test
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Normalizing the data and mapping it between -1 and 1
x_train = ((x_train/255.0) * 2) - 1
x_test = ((x_test/255.0) * 2) - 1

#Checking the shape of the train and test data
print(x_train.shape)
print(x_test.shape)

#Flattening the data
num, height, width = x_train.shape
area = height * width

#Reshaping the data
x_train = np.reshape(x_train, (-1,area))
x_test = np.reshape(x_test, (-1,area))

#Defining the dimensionality of the latent space
latent_dim = 100

#Creating a generator network
def generator_net(latent_dim):

  #Creating a sequential model
  model = Sequential()

  #Adding a dense layer with 256 nodes and a leaky relu activation function
  model.add(Dense(256, input_shape = (latent_dim, ), activation = LeakyReLU(alpha = 0.2)))
  #Adding a batch normalization layer
  model.add(BatchNormalization(momentum = 0.7))
  #Adding a dense layer with 512 nodes and a leaky relu activation function
  model.add(Dense(512, activation = LeakyReLU(alpha = 0.2)))
  #Adding a batch normalization layer
  model.add(BatchNormalization(momentum = 0.7))
  #Adding a dense layer with 1024 nodes and a leaky relu activation function
  model.add(Dense(1024, activation = LeakyReLU(alpha = 0.2)))
  #Adding a batch normalization layer
  model.add(BatchNormalization(momentum = 0.7))
  #Adding a dense layer with 784 nodes and tanh activation function
  model.add(Dense(area, activation = 'tanh'))

  #Returning the model
  return model

#Creating a discriminator network
def discriminator_net(img_size):

  #Creating a sequential model
  model = Sequential()

  #Adding a dense layer with 512 nodes and a leaky relu activation function
  model.add(Dense(512, input_shape = (img_size, ), activation = LeakyReLU(alpha = 0.2)))
  #Adding a dense layer with 256 nodes and a leaky relu activation function
  model.add(Dense(256, activation = LeakyReLU(alpha = 0.2)))
  #Adding a dense layer with 1 node and sigmoid activation function
  model.add(Dense(1, activation = 'sigmoid'))

  #Returning the model
  return model

#Building the discriminator model
discriminator = discriminator_net(area)

#Compiling the discriminator model
discriminator.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = 0.0002, beta_1 = 0.5), metrics = ['accuracy'])

#Building the generator model
generator = generator_net(latent_dim)

#Generating input noise from latent space
z = Input(shape = (latent_dim, ))

#Passing the input noise through the generator to generate an image
img = generator(z)

#Ensuring that only the generator is trained
discriminator.trainable = False

#Labeling fake images as 1 and real ones as 0
fake_pred = discriminator(img)

#Combining the generator and the discriminator
combined_model = Model(z, fake_pred)

#Compiling the combined model
combined_model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = 0.0002, beta_1 = 0.5))

#Defining parameters for training the GAN
batch_size = 32
epochs = 30000
sample_period = 200

#Creating batch labels
ones = np.ones(shape = batch_size)
zeros = np.zeros(shape = batch_size)

#Creating lists to store the respective losses
g_losses = []
d_losses = []

#Creating a folder to store generated images
if not os.path.exists('gan_images'):
  os.makedirs('gan_images')
  
#Creating a function to generate a grid of random samples from the generator
def generate_samples(epoch):

  #Determining the size of the grid
  rows, cols = 5, 5

  #Generating random noise to feed to the generator network
  noise = np.random.randn(rows * cols, latent_dim)

  #Generating random images from the noise
  img = generator.predict(noise)

  #Rescaling the images
  img = 0.5 * img + 0.5

  #Creating the grid
  fig, axs = plt.subplots(rows, cols)

  #Starting from the first index
  index = 0
  
  #Iterating through the grid
  for i in range(rows):
    for j in range(cols):

      #Plotting images in grayscale
      axs[i,j].imshow(img[index].reshape(height, width), cmap='gray')
      axs[i,j].axis('off')
      
      #Incrementing the value of index
      index += 1

  #Saving the grid    
  fig.savefig("gan_images/%d.png" % epoch)
  plt.close()
  
#Starting the training loop
for epoch in range(epochs):
    
  #Selecting a random batch of images
  index = np.random.randint(0, x_train.shape[0], batch_size)
  
  #Obtaining the real images
  real_imgs = x_train[index]
  
  #Generating random noise
  noise = np.random.randn(batch_size, latent_dim)

  #Generating fake images from the noise
  fake_imgs = generator.predict(noise)
  
  #Training the real images on the discriminator network
  d_loss_real, d_acc_real = discriminator.train_on_batch(real_imgs, ones)

  #Training the fake images on the discriminator network
  d_loss_fake, d_acc_fake = discriminator.train_on_batch(fake_imgs, zeros)
  
  #Calculating the loss and the accuracy of the discriminator network
  d_loss = 0.5 * (d_loss_real + d_loss_fake)
  d_acc  = 0.5 * (d_acc_real + d_acc_fake)
  
  #Generating random noise
  noise = np.random.randn(batch_size, latent_dim)
  
  #Calculating the loss of the generator network
  g_loss = combined_model.train_on_batch(noise, ones)
  
  #Repeating the above two steps
  noise = np.random.randn(batch_size, latent_dim)
  g_loss = combined_model.train_on_batch(noise, ones)
  
  #Storing the losses in the respective arrays
  d_losses.append(d_loss)
  g_losses.append(g_loss)
  
  #Printing the loss and accuracy for every 100th epoch
  if epoch % 100 == 0:
    print(f"epoch: {epoch+1}/{epochs}, d_loss: {d_loss:.2f}, d_acc: {d_acc:.2f}, g_loss: {g_loss:.2f}")
  
  #Generate fake images each time the epoch is a multiple of the sample period
  if epoch % sample_period == 0:
    generate_samples(epoch)
    
#Accessing the fake images
!ls gan_images

#Displaying the fake images generated towrds the start of training
from skimage.io import imread
img_grid = imread('gan_images/200.png')
plt.imshow(img_grid)

#Displaying the fake images generated towards the end of training
from skimage.io import imread
img_grid = imread('gan_images/29800.png')
plt.imshow(img_grid)