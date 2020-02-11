# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 13:27:31 2020

@author: Siddhesh
"""

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
from keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, SpatialDropout2D, LeakyReLU, BatchNormalization, Conv2DTranspose, Reshape
from keras.optimizers import Adam

import os
import random

#Specifying the data paths
original_img_dir = "/content/drive/My Drive/Colab Notebooks/Computer Vision/Projects/Pokemon Generator/pokemon_data/dataset"

#Setting the root directory and a data directory
root_dir = os.path.abspath('.')

#Creating a function to preprocess images
def preprocess(image):
    
    #Reading the image
    img = cv2.imread(image)
    
    #Resizing all images to a size of 128 x 128
    img = cv2.resize(img, (128,128))
    
    #Converting images to float32
    img = img.astype('float32')
    
    #Normalizing the image
    img = img/255
    
    #Returning the image
    return img

#Creating a temporary array to store the images
temp = []

#Initializing variable count
count = 1

#Iterating through the training images
for img_dir in glob.glob(original_img_dir + "/*"):
  for pokemon in glob.glob(img_dir + "/*.jpg"):
                 
    #Preprocessing the images
    img = preprocess(pokemon)

    #Storing the preprocessed image in the temporary array
    temp.append(img)

    #Tracking the pokemon and the image count
    print(os.path.basename(img_dir), " ", count, "\n")

    #Incrementing count
    count += 1
        
#Stacking the images
original_img_arr = np.stack(temp)

#Checking the shape of the array of original images
print(original_img_arr.shape)

#Retrieving the size parameters of the image
num, height, width, col = original_img_arr.shape

#Calculating the total number of pixels in the image
volume = height * width * col

#Mapping the pixels between -1 and 1 for the tanh activation function
original_img_arr = (original_img_arr * 2) - 1

#Defining the dimensionality of the latent space
latent_dim = 100

#Creating a generator network
def generator_net(latent_dim):
  
  #Defining the input
  ip1 = Input(shape = (latent_dim, ))
  #Adding a fully connected dense layer with 12288 nodes
  x = Dense(4 * 4 * 256)(ip1)
  #Adding a leaky relu activation function with alpha value 0.2
  x = LeakyReLU(alpha = 0.2)(x)
  #Reshaping the input
  x = Reshape((4, 4, 256))(x)
  
  #Output shape is (None, 4, 4, 256)

  #Adding an inverse convolutional layer with 240 filters and a 5x5 kernel
  x = Conv2DTranspose(240, (5,5), strides = 2, padding = 'same')(x)
  #Adding a leaky relu activation function with alpha value 0.2
  x = LeakyReLU(alpha = 0.2)(x)
  #Adding an inverse convolutional layer with 240 filters and a 5x5 kernel
  x = Conv2DTranspose(240, (5,5), strides = 2, padding = 'same')(x)
  #Adding a leaky relu activation function with alpha value 0.2
  x = LeakyReLU(alpha = 0.2)(x)
  #Adding a batch normalization layer with momentum 0.7
  x = BatchNormalization(momentum = 0.7)(x)

  #Output shape is (None, 16, 16, 240)

  #Adding an inverse convolutional layer with 120 filters and a 3x3 kernel
  x = Conv2DTranspose(120, (3,3), strides = 2, padding = 'same')(x)
  #Adding a leaky relu activation function with alpha value 0.2
  x = LeakyReLU(alpha = 0.2)(x)
  #Adding an inverse convolutional layer with 120 filters and a 3x3 kernel
  x = Conv2DTranspose(120, (3,3), strides = 2, padding = 'same')(x)
  #Adding a leaky relu activation function with alpha value 0.2
  x = LeakyReLU(alpha = 0.2)(x)
  #Adding a batch normalization layer with momentum 0.7
  x = BatchNormalization(momentum = 0.7)(x)

  #Output shape is (None, 64, 64, 120)

  #Creating an output layer with tanh activation
  op1 = Conv2DTranspose(3, (3,3), strides = 2, padding = 'same', activation = 'tanh')(x)
  #Creating a model
  model = Model(ip1, op1)

  #Output shape is (None, 128, 128, 3)

  return model

#Checking summary of the generator
generator_net(latent_dim).summary()

#Creating a discriminator network
def discriminator_net(img):

  #Retrieving the size parameters of the image
  num, height, width, col = img.shape

  #Defining the input
  ip1 = Input(shape = (height, width, col))
  
  #Adding a convolutional layer with 240 filters and a 5x5 kernel
  x = Conv2D(240, (5,5), strides = 2, padding = 'same')(ip1)
  #Adding a leaky relu activation function with alpha value 0.2
  x = LeakyReLU(alpha = 0.2)(x)
  #Adding a convolutional layer with 240 filters and a 5x5 kernel
  x = Conv2D(240, (5,5), strides = 2, padding = 'same')(x)
  #Adding a leaky relu activation function with alpha value 0.2
  x = LeakyReLU(alpha = 0.2)(x)
  #Adding a batch normalization layer with momentum 0.7
  x = BatchNormalization(momentum = 0.7)(x)
  
  #Output shape is (None, 32, 32, 240)

  #Adding a convolutional layer with 120 filters and a 3x3 kernel
  x = Conv2D(120, (3,3), strides = 2, padding = 'same')(x)
  #Adding a leaky relu activation function with alpha value 0.2
  x = LeakyReLU(alpha = 0.2)(x)
  #Adding a convolutional layer with 120 filters and a 3x3 kernel
  x = Conv2D(120, (3,3), strides = 2, padding = 'same')(x)
  #Adding a leaky relu activation function with alpha value 0.2
  x = LeakyReLU(alpha = 0.2)(x)
  #Adding a batch normalization layer with momentum 0.7
  x = BatchNormalization(momentum = 0.7)(x)

  #Output shape is (None, 8, 8, 120)

  #Adding a fully connected dense layer with 12288 nodes
  x = Dense(500)(x)
  #Adding a leaky relu activation function with alpha value 0.2
  x = LeakyReLU(alpha = 0.2)(x)
  #Adding a batch normalization layer with momentum 0.7
  x = BatchNormalization(momentum = 0.7)(x)
  #Flattening the data
  x = Flatten()(x)
  
  #Output shape is (None, 32000)

  #Creating an output layer with sigmoid activation
  op1 = Dense(1, activation = 'sigmoid')(x)
  #Creating a model
  model = Model(ip1, op1)

  #Output shape is (None, 1)

  return model

#Checking summary of the discriminator
discriminator_net(original_img_arr).summary()

#Referred https://www.udemy.com/course/advanced-computer-vision/learn

#Building the discriminator model
discriminator = discriminator_net(original_img_arr)

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
      axs[i,j].imshow(img[index].reshape(height, width, col))
      axs[i,j].axis('off')
      
      #Incrementing the value of index
      index += 1

  #Saving the grid    
  fig.savefig("gan_images/%d.png" % epoch)
  plt.close()
  
  #Starting the training loop
for epoch in range(epochs):
    
  #Selecting a random batch of images
  index = np.random.randint(0, original_img_arr.shape[0], batch_size)
  
  #Obtaining the real images
  real_imgs = original_img_arr[index]
  
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