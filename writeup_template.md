#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I built a sequential model to predict the steering angle. 


####2. Attempts to reduce overfitting in the model

The model works on 80-20 training - validation fit to reduce overfitting. 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).
The loss function was mse. 

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I used a correction factor of 0.2 for left and right cameras. I also cropped unwanted portions from the top and bottom of the training images to reduce noise.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to build a sequential layer by stacking up Convolutional Layers.

My first step was to use a convolution neural network model similar to the NVIDIA Architecture. I thought this model might be appropriate because given the complexity of the problem. I thought it would perform better than simpler architectures like LeNet

In order to gauge how well the model was working, I ran the model in the simulator and checked the how steady it was. During the initial runs the car would wobble a lot and go off road fairly quickly. However as i refined the model by adding left and right data sets and by flipping data. It also help adding layers and tweaking filters and kernel size of each Convolutional layer.



At the end of the process, the vehicle is able to drive autonomously around the track 1 without leaving the road.

####2. Final Model Architecture

My model is a variant of the NVIDIA Architecture. It has 3 5X5 Convolutional Layers, with 24, 36 and 48 filters each. That is followed by two 3X3 convolutional layer with 64 filters each. We use data from all 3 channels. The padding used is VALID. The activation function used is RELU (model.py lines 97 - 101)

The final convolutional layer was flattened. Then I added 3 fully connected layers with 120, 80 and 10 nodes. That was followed by the output layer.


The model uses RELU activation to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 91). 


####3. Creation of the Training Set & Training Process

I used the training data set provided. That was sufficient for me build a good model.


To augment the data sat, I also flipped images and angles. This helped add more data points which helped the training. 

After the collection process, I had 48216 number of data points. 


I finally randomly shuffled the data set and used only 80% of the data to train in each epoch. I used 50 epochs for training the model

I used an adam optimizer so that manually training the learning rate wasn't necessary.
