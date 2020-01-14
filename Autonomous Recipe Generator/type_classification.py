# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 22:10:23 2020

@author: Siddhesh
"""

#Importing the required libraries
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

import random

#Loading the data
data1 = np.array(pd.read_csv("/content/drive/My Drive/Colab Notebooks/Computer Vision/Food 101/labels.txt", header = None))
data2 = np.array(pd.read_csv("/content/drive/My Drive/Colab Notebooks/Computer Vision/Food 101/labels2.txt", header = None))

#Creating empty arrays to store the food name and the food type
food_name = []
food_type = []

#Storing 101000 labels of food name and food type
for _name_, _type_ in zip(np.array(data1), np.array(data2)):
  for i in range(0, 1000):
    food_name.append(_name_)
    food_type.append(_type_)
    
#Converting the labels into a numpy array
food_name = np.array(food_name)
food_type = np.array(food_type)

print(food_name.shape)
print(food_type.shape)

#Creating a function to shuffle data
def shuffle(seed):
  random.seed(seed)
  random.shuffle(food_name)
  random.seed(seed)
  random.shuffle(food_type)
  
#Shuffling the data
shuffle(10)
shuffle(20)
shuffle(10)

#Encoding the labels using One-Hot-Encoding
enc = OneHotEncoder(handle_unknown = 'ignore')
food_name = enc.fit_transform(food_name)

#Building an Extra Trees Classification Model
clf = ExtraTreesClassifier(n_estimators = 100, random_state = 0)
model = clf.fit(food_name, food_type.ravel())

#Creating an empty array to store the result
result = []

#Predicting the test images
prediction = cross_val_predict(model, food_name, food_type.ravel(), cv = 4)

#Iterating through our prediction array and storing each prediction into a list
for i in prediction:
    result.append(i)
    
#Calculating accuracy of the model
accuracy = accuracy_score(food_type, result)
print('Accuracy Is : %g%%' %(accuracy * 100))

#Got an accuracy score of 0.999911 (99.99%)