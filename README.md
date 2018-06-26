# Autonomous Vehicle Behavioral Cloning

![alt text][image11]

---
**Behavioral Cloning Project**

The goals / steps of this project are to:

* Use the simulator to collect data of good human driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the neural network with a training and validation set
* Test that the simulated vehicle successfully drives a lap around track one without leaving the road

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"


[image8]: ./pics/center_2016_12_01_13_30_48_287.jpg "Center Camera Image"
[image9]: ./pics/left_2016_12_01_13_30_48_287.jpg "Left Camera Image"
[image10]: ./pics/right_2016_12_01_13_30_48_287.jpg "Right Camera Image"
[image11]: ./pics/parked_car.png "Autonomous Vehicle"
[image12]: ./pics/simulator.png "Simulator"
[image13]: ./pics/cnn-architecture-624x890.png "Simulator"

[image14]: ./pics/cropped_gray.png "Cropped Grayscaled"
[image15]: ./pics/resized.png "Resized"

[image16]: ./pics/blurred.png "Blurred"
[image17]: ./pics/noisy.png "Noisy"


### This repo includes all required files to run the simulator in autonomous mode

This repo contains the following files:

* model.py - contains the script to create and train the model
* drive.py - for driving the car in autonomous mode
* model.h5 - contains my pre-trained convolution neural network 
* README.md - the file you are reading right now summarizing the results
* data.zip - the dataset which Udacity generoulsy provided for the purpose of validating the architecture
* training_data.zip - the dataset which I painstakingly collected by hand
* output_video.mp4 - video of the car _**safely**_ doing a lap around track 1
* linux_sim.zip - the simulator provided by Udacity for collecting data and driving the vehicle autonomously around the track

#### How to Run the Model

First, download and install the Udacity starter kit for term 1 from here: https://github.com/siqb/CarND-Term1-Starter-Kit.

Next, you can test the pre-trained neural network found in the repo simply by executing the following command:

    python drive.py model.h5

Now just launch the simulator and click "Autonomous Mode" and watch the autonomous magic begin.

![alt text][image12]

### Model Architecture and Training Strategy

#### Design Approach

The overall strategy for deriving a model architecture was to first rebuild the NVIDIA architecture from the bottom up, i.e. one layer at a time, to observe what cumulative role each component played in the end result. Then once I rebuilt the entire architecture, I modified it and fine tuned parameters until I could get the car to drive around the track. Long story short: it wasn't very pretty to do it this way. My approach seems to make sense in theory but perhaps only for those who are endowed with sufficient compute capability in their development environments. You'll read more on this point towards the end of (and throughout!) this README....

I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. To combat the overfitting, I modified the model so that ...

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### Model Architecture

I replicated the NVIDIA architecture with some modifications. As I mentioned above, I built it up backwards, from the last layer to the first which is a painfully slow way of doing it. For reference, here is a diagram of the architecture as published on the NVIDIA blog:

![alt text][image13]

Here's how my implementation looks in code:

    model = Sequential()
    model.add(Convolution2D(24,5,5,subsample=(2, 2), input_shape=(64,64,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(36,5,5,subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(48,3,3,subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1))

The main differences from the published architecture are:

* Use of ReLU
    * The NVIDIA paper doesn't specify what kind of activation function was used as the non-linearity but ReLUs are usually a pretty good bet for most applications. There's a good chance ReLUs were used in the original architecture. 
* Use of batch normalization
    * To increase the stability of a neural network, batch normalization normalizes the output of a previous activation layer by subtracting the batch mean and dividing by the batch standard deviation. Batch normalization is kind of like doing preprocessing at every layer of the network instead of just the input layer. I found (this)[https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c] to be an excellent explanation. The end result is less interdependency of network layers on each other...**less overfitting!**
* Use of dropout
    * Dropout is an algorithm which disables a certain percentage of randomly sampled neurons during each epoch. This means that the optimizer is performing gradient descent on effectively several different neural networks. In the end, the results of these different networks are combined. The end result is less interdependency of neurons on each other...**less overfitting!**

A critical addition that I left out, which in retrospect I was probably errant to exclude, is the max pooling layer. Max pooling allows the network to reduce dimensionality of the data after each convolutional layer. Less trainable parameters means less training time. Less training time means more time for rapid experimentation! More experimentaion means more knowledge and intuition gained. Lesson learned. I will go back and benchmark the difference in training time when using max pooling layers. 

#### Preprocessing

I did not use the Keras ```Lambda``` layer for data normalization or the Keras ```Cropping2D``` for cropping because I implemented a seperate preprocessing function using more traditional techniques to perform these tasks instead. This same preprocessing function is called during training and inference. There may be a performance penalty to using this approach over Keras but I chose this way because I felt more comfortable with it at the time. If I were to go back and redo this part, I would try in Keras and benchmark the performance difference. 

### Creation of the Training Set & Training Process

#### Data Collection

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road as was recommended by the course lectures

To capture good driving behavior, I first recorded two laps on track one using center lane driving.

The data collected by driving around the track is biased towards driving straight. This is evidenced by the fact that the car often doesn't turn enough (or sometimes not at all) to avoid driving through the curve and veering off the track. This is especially true of the biggest curves in the track. We need the car to be able to generalize to more driving conditions. I took the following steps to rebalance the dataset:

1. Used the mouse instead of the keyboard for steering. The problem with the keyboard is that the left and right keys for steering are either pressed or they are not. This results in a data set which is fully of mostly zero degree steering angles (our y-labels) with very occassional higher steering angles thrown in. Using the mouse to drive the car creates a more natural experience where the driver has to maintain control over the steering angle at all times. This results in a continuous range of steering angles.
2. Filter out a high percentage of straight driving steering angles. This took some experimentation but I found that filtering out 75% of steering angles less than 0.5 degrees worked.
3. Drive a couple of laps with all "recovery" data. This means that I would drive the car towards the edge of the track, hit record, pull away from the edge of the track with a large steering angle, and then stop recording. The idea is to give the CNN examples of what to do in case the vehicle begins veering off the road since there aren't any examples of this scenario in normal straight biased driving. These images show what a recovery looks like starting from ... :
4. and put 20% of the data into a validation set. 

The car has three camera...one in the center, one on the left, and one on the right. Here is an example of what the car sees from all three cameras.

![alt text][image8]

![alt text][image9]

![alt text][image10]


I did **not** repeat this process on track two in order to get more data points. I found the data I collected from track 1 to be sufficient for the purpose of driving a lap around track 1. To expand the data set, I repeated some augmentation techniques from Lab 2. lsAfter the collection process, I had many data points. I then preprocessed this data by running it through a short pipeline:

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

#### CNN Training

The way Keras works, ```fit_generator()``` calls a user defined generator function which yields the training samples on a batch by batch basis until all of the training data is exhausted. The generator pipeline has several basic functions:

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
    2. I also repeated some techniques from the traffic sign classfier lab (lab 2). I added a gaussian blurred and noisy versions of each image. I am not sure how quantifiably effective these methods were but it seems like I got enough good data.
    3. I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:
    4. There are many more augmentation techniques which could have been used. However, you don't want to have a dataset larger than what you need to accomplish your task because of the increased time and complexity. I found that with these simple trechniques alone, I was successfully able to pull off a lap around the track. I did not attempt to drive around the challenge track but I know that it has varying lighting conditions that track 1 does not have. If I wanted the car to be able to drive around the challenge track, it would be a good idea to augment the training data with varying lighting conditions like I did in the traffic sign classifier project.
    
4. I finally randomly shuffled the data batch before yielding it back to the Keras fit_generator. 

Here's an example of some preprocessed images.

This is how it looks when the image has been grayscaled and cropped to the correct ROI showing only the road.

![alt text][image14]

This is how it looks when the same image has been resized to the input size of the neural network.

![alt text][image15]

Here's an example of some augmented images.

This is an example of the same image flipped horizontally.

![alt text][image16]

This is an example of the same image blurred.

![alt text][image17]

This is an example of the same image with noise.

![alt text][image18]

There is a lesson in this...I feel that I actually did a really **bad** job with data augmentation! The reason I say this is because at the time when I was learning, I dind't realize that data augmentation isn't purely for the sake of adding more data...it is for adding more **relevant** data! Blurred and noisy images are probably irrelevant in this context because there is no scenario in our simulator (at leats in track 1, not sure about track 2) in which the car will encounter blurry or noisy conditions. Yes, in real life, these would be very good data augmentations because blurring and noise can and will occur due to bad weather, bad equipment, etc. But not in this simulator!

That being said, the reason I am keeping this in my report is to remind me of this critical lesson in data collection! Maybe one day I will go back and augment the data more appropriately and ocmpare the difference in performance.

#### Overfitting

To reduce overfitting, I used dropout. I started with a keep probability of 50% but through experimentation, I dropped it to 10%. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### Parameter Tuning

I was able to pull off a successful lap using only five epochs to train. It's possible that more epochs could have lead to better driving behavior but this comes at a very steep cost: increased training time. A few minutes extra may not sound like much, but I already sunk massive amounts of time fine tuning and testing the CNN. Between work and family, time is already at a premium. Shaving off even a few minutes of training time compounds into several hours of savings in the long run. Again, knowing how to balance trade-offs is part of the art of deep learning and this was the ideal lesson to teach me that.

The model used an adam optimizer, so the learning rate was not tuned manually. I used an adam optimizer so that manually training the learning rate wasn't necessary. I read about ways to play with the learning rate even while using the Adam optimizer and at one point was tempted to do so. But I decided that my time was better spent pursuing bigger gains. Alas, I have learned that knowing when to make such judgement calls is part of the art of deep learning.

#### Training Time

As alluded to several times throughout this write up, the main problem I had with this lab was the inability to do fast iterations of experiments. It is well known that deep learning is largely experimental...but it's hard to experiment if it takes too long to train!

My hardware is a four year old laptop with Intel Core i7 CPU and an NVIDIA GeForce GT 740M GPU running Ubuntu 16.04. I am not a gamer so this computer has served me well...until now. If I recall correctly, this laptop costed me somewhere in the range of $700 - $800. Clearly, anything in this price range is going to be relatively underpowered and can't be used for any heavy lifting. Even the highest end gaming laptops are underpowered compared to the computing potential of a desktop system. Although I don't have the $$$ to burn on a fancy new system just for running neural networks, I just got a fancy new desktop system with a heavy duty GPU at work that I can tinker around with. I am hoping to go back and fine tune this lab on my work computer when I get a chance. I hope to update this repo with results! 

In the beginning when I was first experimenting with CNN architectures, I recall that a single epoch could take 15+ minutes meaning that just a few epochs would take well over an hour! As I began to refine the model, I _eventually_ got training time to about three or four minutes per epoch. However, this is still not fast enough for rapid experimentation and tuning of parameters.

I tried experimenting with the batch size to speed up training. I pushed it all the way up to 128 but did not see a noticeable performance increase. When I pushed it beyond 128 (i.e. 256), my GPU ran out of memory. I also noticed that if I had Jupyter notebooks running in the background (i.e. lab 2), then the GPU would also run out of memory regardless of batch size, even if the notebook wasn't actively executing. I guess the notebook was just holding allocated memory and not letting it go. 

