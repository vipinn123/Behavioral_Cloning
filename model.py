# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 11:00:21 2017

@author: vipin
"""
from PIL import Image
import numpy as np
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

lines = []

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    
    for line in reader:
        lines.append(line)

images = []
measurements = []

#correction factor for the left and right camera
correction = 0.2

for line in lines[1:]:
    center_img_source_path = line[0]
    center_img_filename = center_img_source_path.split('/')[-1]
    
    left_img_source_path = line[1]
    left_img_filename = left_img_source_path.split('/')[-1]
    
    right_img_source_path = line[2]
    right_img_filename = right_img_source_path.split('/')[-1]
    
    folder_path = './data/IMG/'
    
    #get the center, left and right camera images
    img_center = np.asarray(Image.open(folder_path + center_img_filename))
    img_left = np.asarray(Image.open(folder_path + left_img_filename))
    img_right = np.asarray(Image.open(folder_path + right_img_filename))
    
    #flip images for data augmentation
    img_center_flipped = np.fliplr(img_center)
    img_left_flipped = np.fliplr(img_left)
    img_right_flipped = np.fliplr(img_right)
    

    #add the images to the processing pipeline
    images.append(img_center)
    images.append(img_center_flipped)
    images.append(img_left)
    images.append(img_left_flipped)
    images.append(img_right)
    images.append(img_right_flipped)
    
    #add the target steering angle values for each of the above images
    measurement = float(line[3])
    measurements.append(measurement)
    center_measurement_flipped = -measurement
    measurements.append(center_measurement_flipped)
    
    measurements.append(measurement+correction)
    left_measurement_flipped = -measurement-correction
    measurements.append(left_measurement_flipped)
    
    measurements.append(measurement-correction)
    right_measurement_flipped = -measurement + correction
    measurements.append(right_measurement_flipped)


#training feature set
X_train = np.array(images).reshape((48216,160,320,3))

#training target values
Y_train = np.array(measurements)


   

from keras.models import Sequential
from keras.layers import Flatten, Dense,Lambda
from keras.layers import Convolution2D,MaxPooling2D,Cropping2D

#build a sequential model to predict the continuous steering angle values
model = Sequential()

#normalize the pixel values for mean = 0
model.add(Lambda(lambda X_train: (X_train / 255.0) - 0.5, input_shape=(160,320,3)))

#crop unwanted portions of the frame to remove noise
model.add(Cropping2D(cropping=((70,25), (0,0))))

#build the network modeled around NVIDIA Architecture
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', dim_ordering='tf',activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', dim_ordering='tf',activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', dim_ordering='tf',activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3,3, activation="relu"))


model.add(Flatten())
model.add(Dense(120))
model.add(Dense(80))
model.add(Dense(10))
model.add(Dense(1))

#minimise mean square error
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_train, validation_split=0.2,shuffle=True,nb_epoch=50)

model.save('model.h5')

    
    
    