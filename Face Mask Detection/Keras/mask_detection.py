#!/usr/bin/env python
# coding: utf-8

# ### Face Mask Detection

# ##### Referred https://www.youtube.com/watch?v=d3DJqucOq4g and https://github.com/aieml/face-mask-detection-keras

# In[1]:


#Importing required libraries
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model


# In[2]:


#Changing the directory
os.chdir("C:/Users/Siddhesh/Desktop/Siddhesh/Projects/Face Mask Detection/Project/Data")


# In[3]:


#Loading the Neural Network model trained earlier
model = load_model("model-009.model")

#Loading the face classifier
face_cls = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Accessing the default webcam
source = cv2.VideoCapture(0)

#Creating a dictionary of labels
labels_dict = {0 : 'Wearing Mask', 1 : 'Not Wearing Mask'}

#Creating a dictionary of colors (green and blue)
color_dict = {0 : (0,255,0), 1: (0,0,255)}


# In[ ]:


#Running indefinitely
while(True):
    
    #Accessing the image from the webcam
    ret, img = source.read()
    
    #Converting the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #Detecting face in the image
    faces = face_cls.detectMultiScale(gray_img, 1.3, 5)
    
    #Extracting the coordinates of the pixels of the face
    for x, y, w, h in faces:
        
        #Defining the region of interest
        face_img = gray_img[y:y+w, x:x+w]
        
        #Resizing the image to the size of the trained data
        resized_img = cv2.resize(face_img, (100,100))
        
        #Normalizing the image
        normalized_img = resized_img/255
        
        #Reshaping the image to indicate one grayscale image
        reshaped_img = np.reshape(normalized_img, (1,100,100,1))
        
        #Predicting the result
        result = model.predict(reshaped_img)
        
        #Extracting the label from the softmax probabilities
        label = np.argmax(result, axis = 1)[0]
        
        #Constructing the bounding boxes
        cv2.rectangle(img, (x,y), (x+w, y+h), color_dict[label], 2)
        cv2.rectangle(img, (x, y-40), (x+w, y), color_dict[label], -1)
        cv2.putText(img, text = labels_dict[label], org = (x,y-10), fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = 0.8, color = (255,255,255), thickness = 2)
        
    #Showing the live image
    cv2.imshow("LIVE", img)
    
    #Defining the wait time
    key = cv2.waitKey(delay = 1)
    
    #Defining a key to close the video
    if(key == 8):
        
        #Breaking the loop
        break
        
#Destroying all windows
cv2.destroyAllWindows()

#Closing the webcam
source.release()

