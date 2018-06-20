# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md (the file you are reading right now) summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

I followed the NVIDIA arhcitecture

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.



#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road as was recommended by the course lectures. T 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

##### Data Collection

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]


The data collected by driving around the track is biased towards driving straight. This is evidenced by the fact that the car often doesn't turn enough (or sometimes not at all) to avoid driving through the curve and veering off the track. This is especially true of the biggest curves in the track. We need the car to be able to generalize to more driving conditions. I took the following steps to rebalance the dataset:

1. Used the mouse instead of the keyboard for steering. The problem with the keyboard is that the left and right keys for steering are either pressed or they are not. This results in a data set which is fully of mostly zero degree steering angles (our y-labels) with very occassional higher steering angles thrown in. Using the mouse to drive the car creates a more natural experience where the driver has to maintain control over the steering angle at all times. This results in a continuous range of steering angles.
2. Filter out a high percentage of straight driving steering angles. This took some experimentation but I found that filtering out 75% of steering angles less than 0.5 degrees worked.
3. Drive a couple of laps with all "recovery" data. This means that I would drive the car towards the edge of the track, hit record, pull away from the edge of the track with a large steering angle, and then stop recording. The idea is to give the CNN examples of what to do in case the vehicle begins veering off the road since there aren't any examples of this scenario in normal straight biased driving. These images show what a recovery looks like starting from ... :
4. and put 20% of the data into a validation set. 

![alt text][image3]
![alt text][image4]
![alt text][image5]


I did not repeat this process on track two in order to get more data points. I found the data I collected from track 1 to be sufficient. To expand the data set, I repeated some augmentation techniques from Lab 2. lsAfter the collection process, I had many data points. I then preprocessed this data by running it through a short pipeline:

##### CNN Training

The way Keras works, the ```fit_generator``` calls a generator function which yields the training samples on a batch by batch basis until all of the training data is exhausted. The generator pipeline has several basic functions:

1. Loads training samples for all three cameras by the batch.
2. Preprocess the image by running it through a short preprocessing pipeline:

    1. Cropping down the image in the rows (y) direction. I tried many different ROIs. At first, I wanted to also crop in the cols (x) direction but very time I tried to do so, a significant part of the lane lines would get cut out. Lane lines are the primary feature that our neural network uses to learn to drive so I opted against trying to crop in this direction.
    2. Converted color space from RGB to grayscale. None of the features that our CNN uses to learn to drive are dependent on color. Therefore, it is best to remove the extra color channels to reduce input dimensionality.
    3. Normalized for zero mean and unit variance.
    4. Resizing the image to 64x64 pixels. I chose this size because it was close to the image size used in the NVIDIA architecture. 
    5. **NOTE:** The exact same preprocessing pipeline must be used during inference! Therefore, I modified ```drive.py``` to also call the same code, ```preprocess()```, in real-time for each frame broadcast from the simulator. Because of this, I did not use the Keras API for cropping.

2. For the left and right cameras, there is no y-label provided in the driving log CSV file. Therefore, we have to create our own label by adding (for left camera) or subtracting (fro right camera) a fixed correction to the steering angle from the ceneter camera. This involved a lot of experimentation but I ultimately found a steering correctiong of 0.30 to work well for me.
3. Augment each training image in the batch
    1. The most important thing I did was to flip my images and angles and append them to the data set. The great thing about this approach is that it pretty much replicates what the vehicle would see if it was driving on the track in the opposite direction.

  ![alt text][image6]
  ![alt text][image7]

    2. I also repeated some techniques from the traffic sign classfier lab (lab 2). I added a gaussian blurred and noisy versions of each image. I am not sure how quantifiably effective these methods were but it seems like I got enough good data.
    3. I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:
    4. There are many more augmentation techniques which could have been used. However, you don't want to have a dataset larger than what you need to accomplish your task because of the increased time and complexity. I found that with these simple trechniques alone, I was successfully able to pull off a lap around the track. I did not attempt to drive around the challenge track but I know that it has varying lighting conditions that track 1 does not have. If I wanted the car to be able to drive around the challenge track, it would be a good idea to augment the training data with varying lighting conditions like I did in the traffic sign classifier project.
    
4. I finally randomly shuffled the data batch before yielding it back to the Keras fit_generator. 

To reduce overfitting, I used dropout. I started with a keep probability of 50% but through experimentation, I dropped it to 40%. 

I used an adam optimizer so that manually training the learning rate wasn't necessary. I read about ways to play with the learning rate even while using the Adam optimizer and at one point was tempted to do so. But I decided that my time was better spent pursuing bigger gains. Alas, I have learned that knowing when to make such judgement calls is part of the art of deep learning.

I was able to pull off a successful lap using only five epochs to train. It's possible that more epochs could have lead to better driving behavior but this comes at a very steep cost: increased training time. A few minutes extra may not sound like much, but I already sunk massive amounts of time fine tuning and testing the CNN. Between work and family, time is already at a premium. Shaving off even a few minutes of training time compounds into several hours of savings in the long run. Again, know how to balance trade-offs is part of the art of deep learning and this was the ideal lesson to teach me that.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... 





