# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 12:43:04 2019

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
img_path = "/content/drive/My Drive/Colab Notebooks/Computer Vision/Self Driving Car/Road Lanes/Image/test_image.jpg"
vid_path = "/content/drive/My Drive/Colab Notebooks/Computer Vision/Self Driving Car/Video/test2.mp4"

#Creating a function to convert an image into its gradient
def to_gradient(image):

    #Converting the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    #Reducing noice by replacing plixel values with the average of values of pixels around it
    blurred_image = cv2.GaussianBlur(gray_image, (5,5), 0)

    #Creating a gradient image by measuring the deviation in pixels
    gradient_image = cv2.Canny(blurred_image, 50, 150)
    
    #Returning the gradient
    return gradient_image

#Creating a function to isolate the region of interest
def region_of_interest(image):
    
    #Finding the height of the image
    height = image.shape[0]
    
    #Visualizing the coordinates of the the region we want to isolate
    #plt.imshow(image)
    
    #Creating a polygon with the same coordinates as the region of interest
    polygon = np.array([[(200, height),(1100, height),(550, 250)]])
    
    #Creating a black mask with the same size as that of the image
    mask = np.zeros_like(image)
    
    #Filling the black mask with the region of interest
    region = cv2.fillPoly(mask, polygon, 255)
    
    #Using the black mask to display only the required region of the original image
    required_image = cv2.bitwise_and(image, mask)
    
    #Returning the region of interest
    return required_image

#Creating a function to display lines in the black masked space
def display_lines(image, lines):
    
    #Creating a black mask with the same size as that of the image
    mask = np.zeros_like(image)
    
    #Checking whether lines is not empty
    if(lines is not None):
        
        #Retreiving the coordinates of each line
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            
            #Drawing a blue line of thickness 10 with these coordinates in the black mask
            cv2.line(mask, (x1,y1), (x2,y2), (255,0,0), 10)
            
    #Returning the black masked region
    return mask

#Creating a function to define coordinates
def define_coordinates(image, line_parameters):
    
    #Retreiving the slope and intercept of the line
    slope, intercept = line_parameters
    
    #Retrieving the coordinates of the line
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    
    #Returning the four coordinates
    return np.array([x1, y1, x2, y2])

#Creating a function to compute slope ad intercept of a line
def average_slope_intercept(image, lines):
    
    #Creating two arrays to store lines with negative and positive slopes separately
    left_fit = []
    right_fit = []
    
    #Checking whether lines is not empty
    if(lines is not None):
        
        #Retreiving the coordinates of each line
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            
            #Computing the slope and intercept of each line
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            
            #Storing the parameters in their designated arrays
            if(slope < 0):
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
                
        #Computing the average slope-intercept tuple of the left lane and right lane
        left_fit_avg = np.average(left_fit, axis = 0)
        right_fit_avg = np.average(right_fit, axis = 0)
        
        #Retrieving coordinates of the left and right line
        left_line = define_coordinates(image, left_fit_avg)
        right_line = define_coordinates(image, right_fit_avg)
        
        #Returning the two lines
        return np.array([left_line, right_line])
    
# ****************************** Testing whether detected in image **************************************************    
    
#Reading the image
image = cv2.imread(img_path)

#Creating a copy of the image so that there are no changes in the original image
lane_image = np.copy(image)

#Converting the lane image into a gradient
gradient_image = to_gradient(lane_image)

#Retrieving the region of interest
masked_image = region_of_interest(gradient_image)

#Detecting lines in Hough Space with a precision of 2 pixels and 1 degree. Reject lines with length less than 40
lines = cv2.HoughLinesP(masked_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)

#Retrieving the left and right lines
averaged_lines = average_slope_intercept(lane_image, lines)

#Retrieving the region with lines
line_image = display_lines(lane_image, averaged_lines)

#Blending the lines with the lane image and reducing the intensity of the lane image to make the lines more prominent
blended_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

#Displaying the image
cv2_imshow(blended_image)
cv2.waitKey(0)

#Reading the image
image = cv2.imread(img_path)

#Creating a copy of the image so that there are no changes in the original image
lane_image = np.copy(image)

#Converting the lane image into a gradient
gradient_image = to_gradient(lane_image)

#Retrieving the region of interest
masked_image = region_of_interest(gradient_image)

#Detecting lines in Hough Space with a precision of 2 pixels and 1 degree. Reject lines with length less than 40
lines = cv2.HoughLinesP(masked_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)

#Retrieving the left and right lines
averaged_lines = average_slope_intercept(lane_image, lines)

#Retrieving the region with lines
line_image = display_lines(lane_image, averaged_lines)

#Blending the lines with the lane image and reducing the intensity of the lane image to make the lines more prominent
blended_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

#Displaying the image
cv2_imshow(blended_image)
cv2.waitKey(0)


# ****************************** Testing whether detected in video **************************************************    

#Creating a video capturing object
cap = cv2.VideoCapture(vid_path)

while(cap.isOpened()):
    _, frame = cap.read()
    
    #Converting the lane image into a gradient
    gradient_image = to_gradient(frame)

    #Retrieving the region of interest
    masked_image = region_of_interest(gradient_image)

    #Detecting lines in Hough Space with a precision of 2 pixels and 1 degree. Reject lines with length less than 40
    lines = cv2.HoughLinesP(masked_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)

    #Retrieving the left and right lines
    averaged_lines = average_slope_intercept(frame, lines)

    #Retrieving the region with lines
    line_image = display_lines(frame, averaged_lines)

    #Blending the lines with the lane image and reducing the intensity of the lane image to make the lines more prominent
    blended_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    
    #Displaying the image
    cv2_imshow(blended_image)
    
    #Breaking the loop upon pressing 'q'
    if(cv2.waitKey(1) == ord('q')):
        break
        
cap.release()
cv2.destroyAllWindows()

