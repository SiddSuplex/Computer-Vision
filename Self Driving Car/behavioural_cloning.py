# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 02:39:25 2019

@author: Siddhesh
"""

#Referred the tutorial offered at https://www.udemy.com/course/applied-deep-learningtm-the-complete-self-driving-car-course

#Retreiving the data file
#!git clone https://github.com/rslim087a/track.git

#Checking the contents in the data file
#!ls track

#Specifying the data directory
data_directory = 'track'

#Specifying the columns
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']

#Storing the driving logs in a variable
logs = pd.read_csv(os.path.join(data_directory, 'driving_log.csv'), names = columns)

#Ensuring that the name of the data doesn't overflow
pd.set_option('display.max_colwidth', -1)

#Creating a function to split the long path
def split_path(path):
  
  #Splitting the head and tail of the path
  head, tail = ntpath.split(path)
  
  #Returning the tail of the path
  return tail

#Replacing the long path with just its tail for every image
logs['center'] = logs['center'].apply(split_path)
logs['left'] = logs['left'].apply(split_path)
logs['right'] = logs['right'].apply(split_path)

logs.head(5)

#Setting an odd number of bins in order to get a proper median(center)
num_bins = 25

#Setting a particular threshold in order to remove bias towards a particular steering angle
threshold = 400

#Finding the deviation of the steering from the center
hist, bins = np.histogram(logs['steering'], num_bins)
#print(bins)

#Centering the steering at an angle of 0 in order to ensure the vehicle drives straight
center = (bins[:-1] + bins[1:]) * 0.5
#print(center)

#Plotting the histogram
plt.bar(center, hist, width = 0.05)

print(len(logs["steering"]))

#Inititalizing an empty list for the steering data to be removed
remove_list = []

for i in range(0, num_bins):
  #Creating a temporary list to store the steering data
  temp = []
  
  for j in range(0, len(logs["steering"])):
    
    #Checking if the steering data belongs to a particular bin   
    if(logs["steering"][j] >= bins[i] and logs["steering"][j] <= bins[i+1]):
      
      #Storing the steering data
      temp.append(j)
    
  #Shuffling the steering data in order to preserve information even after eliminating some data
  temp = shuffle(temp)
  
  #Filtering out the unwanted data
  temp = temp[threshold:]
  
  #Storing the unwanted data in the remove list
  remove_list.extend(temp)
    
#print(len(remove_list))

#Dropping the data the steering data to be removed from the steering data
logs.drop(logs.index[remove_list], inplace = True)
#print(len(logs))

#Finding the deviation of the steering from the center
hist, flag = np.histogram(logs['steering'], num_bins)

#Plotting the histogram
plt.bar(center, hist, width = 0.05)

#Loading the steering image data
def load_image(data_directory, data):
  img_paths = []
  steering_angles = []
  
  for i in range(0, len(data)):
    #Accessing data at the ith row    
    indexed_data = data.iloc[i]
    
    #Retreiving the file names of images of a particular instance taken from the center, left and right
    center = indexed_data[0]
    left = indexed_data[1]
    right = indexed_data[2]
    
    #Eliminating whitespaces and retreiving the paths of images of a particular instance taken from the center
    img_paths.append(os.path.join(data_directory, center.strip()))
    
    #Retreiving the steering angles
    steering_angles.append(indexed_data[3])
  
  #Converting the arrays into numpy arrays
  img_paths = np.asarray(img_paths)
  steering_angles = np.asarray(steering_angles)
  
  #Returning the image paths and the steering angles
  return img_paths, steering_angles


#Retreiving the image paths and the steering angles
img_paths, steering_angles = load_image(data_directory + '/IMG', logs)

#Splitting data into training and validation data
x_train, x_val, y_train, y_val = train_test_split(img_paths, steering_angles, test_size = 0.2, random_state = 6)

#Creating a function to preprocess images
def img_preprocess(img):

  #Modifying the height of hte preprocessed image
  img = img[60:135,:,:]
  
  #Converting image to YUV format for Nvidia Model
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
  
  #Removing noise from the image using a 3x3 kernel with 0 deviation
  img = cv2.GaussianBlur(img, (3,3), 0)
  
  #Resizing the image
  img = cv2.resize(img, (200,66))
  
  #Normalizing the image
  img = img/255
  
  #Returning the image
  return img

#Creating an image generator
def image_generator(img_paths, steering_angles, batch_size, is_training):

  #Creating a function to create random augmented images
  def randomize_augment(image, steering_angle):
    
    #Creating a function to zoom an image
    def zoom(image):
      #Specifying the range of the zoom to be 30%
      zoom = aug.Affine(scale = (1.0, 1.3))

      #Zooming the image
      image = zoom.augment_image(image)

      #Returning the zoomed image
      return image

    #Creating a function to pan an image
    def pan(image):
      #Specifying the range of the pan to be 10% left, 10% right, 10% down and 10% up
      pan = aug.Affine(translate_percent = {"x" : (-0.1,0.1), "y" : (-0.1,0.1)})

      #Panning the image
      image = pan.augment_image(image)

      #Returning the panned image
      return image

    #Creating a function to alter brightness in an image
    def brightness(image):
      #Specifying the range of the value with which pixels are to be multiplied
      brightness = aug.Multiply(mul = (0.2, 1.2))

      #Altering the image brightness
      image = brightness.augment_image(image)

      #Returning the modified image
      return image

    #Creating a function to flip the image by a particular angle
    def flip(image, steering_angle):
      #Flipping the image
      image = cv2.flip(image, 1)

      #Flipping the negative angles with the positive angles
      steering_angle = -steering_angle

      #Returning the flipped image with its steering angle
      return image, steering_angle

    #Randomizing zoom
    if(np.random.rand() < 0.5):
      image = zoom(image)

    #Randomizing pan
    if(np.random.rand() < 0.5):
      image = pan(image)

    #Randomizing brightness
    if(np.random.rand() < 0.5):
      image = brightness(image)

    #Randomizing flip
    if(np.random.rand() < 0.5):
      image, steering_angle = flip(image, steering_angle)

    #Returning the modified image with its steering angle
    return image, steering_angle

  while(True):
    #Creating empty list to store the images and steering angles in the form of batches
    batch_img_list = []
    batch_steering_list = []

    for i in range(0,  batch_size):
      #Generating a random index
      random_index = random.randint(0, (len(img_paths) - 1))

      #Checking whether the data belongs to the training data
      if(is_training):
        img, angle = randomize_augment(img_paths[random_index], steering_angles[random_index])
      else:
        #Reading the image
        img = mplimg.imread(img_paths[random_index])
        angle = steering_angles[random_index]

    #Preprocessing the image
    img = img_preprocess(img)

    #Appending the batches
    batch_img_list.append(img)
    batch_steering_list.append(img)

  #Converting the batches into numpy arrays
  batch_img_list = np.asarray(batch_img_list)
  batch_steering_list = np.asarray(batch_steering_list)

  yield(batch_img_list, batch_steering_list)

#Creating augmented training and validation data
x_train_gen, y_train_gen = next(image_generator(x_train, y_train, 1, 1))
x_val_gen, y_val_gen = next(image_generator(x_val, y_val, 1, 1))

#Creating an NVIDIA Neural Network
def nvidia_model(LR):
  #Creating a sequential model
  model = Sequential()

  #Adding a Convolutional Neural Network with 24 filters and a 5x5 kernel window that moves 2 pixels at a time instead of 1
  model.add(Convolution2D(24, 5, 5, subsample = (2,2), input_shape = (66, 200, 3), activation = 'elu'))

  #Adding a Convolutional Neural Network with 36 filters and a 5x5 kernel window that moves 2 pixels at a time instead of 1
  model.add(Convolution2D(36, 5, 5, subsample = (2,2), activation = 'elu'))

  #Adding a Convolutional Neural Network with 48 filters and a 3x3 kernel window that moves 2 pixels at a time instead of 1
  model.add(Convolution2D(48, 3, 3, subsample = (2,2), activation = 'elu'))
  
  #Adding a Convolutional Neural Network with 64 filters and a 3x3 kernel window
  model.add(Convolution2D(64, 3, 3, activation = 'elu'))
  
  #Adding a Convolutional Neural Network with 64 filters and a 3x3 kernel window
  model.add(Convolution2D(64, 3, 3, activation = 'elu'))

  #Adding a Dropout Layer to prevent overfitting
  model.add(Dropout(0.5))

  #Flattenning the Neural Network
  model.add(Flatten())

  #Adding a Fully Connected Layer
  model.add(Dense(100, activation = 'elu'))

  #Adding a Dropout Layer to prevent overfitting
  model.add(Dropout(0.5))

  #Adding a Fully Connected Layer
  model.add(Dense(50, activation = 'elu'))

  #Adding a Dropout Layer to prevent overfitting
  model.add(Dropout(0.5))

  #Adding a Fully Connected Layer
  model.add(Dense(10, activation = 'elu'))

  #Adding a Dropout Layer to prevent overfitting
  model.add(Dropout(0.5))
  
  #Adding an Output Layer that performs regression
  model.add(Dense(1))

  #Compiling the model
  model.compile(optimizer = Adam(lr = LR), loss = 'mse')

  #Returning the model
  return model

#Checking the summary of the model
model = nvidia_model(0.001)
model.summary()

#Fitting the augmented training and validation data to the model
history = model.fit_generator(image_generator(X_train, y_train, 100, 1), steps_per_epoch=300, epochs=10, validation_data=batch_generator(X_valid, y_valid, 100, 0), validation_steps=200, verbose=1, shuffle = 1)

#Plotting training loss vs validation loss to detect overfitting
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])

