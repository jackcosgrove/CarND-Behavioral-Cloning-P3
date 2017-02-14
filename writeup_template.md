#**Behavioral Cloning Project** 

###Jack Cosgrove
###February 13, 2017

---

###The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolutional neural network (CNN) in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around Track 1 without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/augmentation.png "Image Augmentation"
[image2]: ./examples/udacity_hist.png "Udacity Steering Histogram"
[image3]: ./examples/forward_hist.png "Forward Steering Histogram"
[image4]: ./examples/reverse_hist.png "Reverse Steering Histogram"

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
* testing.mp4 video of a successful run around Track 1

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of a convolutional neural network with 3x3 filter sizes and depths between 32, 64, and 128 (model.py lines 252-262) Each convolutional layer is followed by a max pooling layer with a 2x2 filter size. These are then followed by a dropout layer which drops out 50% of the features.

The visual features are flattened and passed to three fully-connected layers (code line 266-268). These use RELU activation to introduce nonlinearity. The layer sizes are 512, 64, and 16, gradually narrowing the pipeline into a single output feature.

Lastly there is a fully-connected layer with one output and a linear activation to procude the steering angle (code line 273).

I did not scale or normalize my images in the CNN pipeline since I also crop the images, requiring edits to drive.py. I decided to keep these transformations in the generator pipeline since so many other transformations were included there.

I see now that the course materials have been updated to show how to crop, scale, and normalize in Keras. However I have limited access to a training GPU and will have to submit my pipeline with the transformations in the generator and drive.py.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 254, 258, and 262). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 285). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. See the video testing.mp4.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 278).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I started with the Udacity data set. I then used a combination of center lane driving, recovering from the left and right sides of the road, and driving the course in reverse to equalize the frequency of left and right turns. I also repeatedly drove troublesome sections of the course, toggling recording so I could reverse and cover the same ground again.

I smoothed the steering angle data as part of the preprocessing to compensate for the often jerky control devices. I did not use the keyboard to control the car while training; I used a game controller. I found experimentally that a running average window of 4 samples produced the smoothest results.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use known models as a starting point. I knew that I had to use convolutional layers before fully connected layers to extract visual features that are unknown a priori. I began with the Nvidia architecture, and expanded this to be more similar to the architecture shown [here](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.p5th2guac). My final architecture is simpler, and does not include ELU layers for example.

My first step was to use a CNN model similar to the Nvidia architecture. I thought this model might be appropriate because it was proven to work and was relatively simple. After adding the preprocessing techniques shown in the link above, I adapted my architecture to be closer to what was described. Most importantly, this included adding max pooling layers, which are useful when filtering out the dead spaces created by translating the images, and moving the dropout layers after the convolutional layers, rather than after the fully connected layers.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Because I had included dropout layers in the model to combat overfitting from the start, I obtained similar losses between my training and validation sets. 

Initially I had poor success and could not get past the first curve. Some architectures would always output the same steering angle, which meant the data was imbalanced and the network had learned to minimize loss by optimizing for the most common case of driving straight. After different changes to my architecture and similar poor results, I realized that training data quality was the most likely cause of my failures rather than a poor architecture. Once I was able to achieve variable steering angle output using my architecture and a small amount of data, I left the architecture as-is and concentrated on data balancing and augmentation.

I had particular trouble with the right turn alongside the lake. I drove this section repeatedly in forward and reverse to learn how to deal with this turn.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

Amazingly, my model made it all the way to the top of the second course on its first try!

####2. Final Model Architecture

The final model architecture (model.py lines 252-273) is the same as what is described above. Once I had good data the model worked fine.

####3. Creation of the Training Set & Training Process

I started using the Udacity data set. I plotted the steering angles in a histogram and noticed that while the data were evenly distributed, there were few outliers. Simulations trained using the Udacity data set would consistently miss the first curve. The car would not turn hard enough because the steering angles in the data set were too soft.

![Figure 1][image2]

To capture more extreme driving behavior, I first recorded three laps on track one using center lane driving. The outliers were introduced by my game controller's sensitivity and my own gracelessness. I plotted a histogram of this data and noticed that it favored left turns, which is understandable given the counterclockwise driving direction.

![Figure 2][image3]

I then recorded two laps driving the course in reverse to balance out the steering angle.

![Figure 3][image4]

I tried slaloming through the course and removing the parts of each slalom where I drifted from the center to the edge to train the network to correct course, however this data was bad and caused worse performance. In the end, my own clumsiness while driving introduced enough corrective data. The drifting from the center was more than offset by the majority of data points where I stayed in the center, while the steering angles at the edge of the road were mostly corrective.

I never trained on track two.

To augment the data set, I used a number of techniques described [here](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.p5th2guac). The brightness adjustments and shadow casting proved critical for driving the second course.

Here are examples of the various augmentations I used, along with a composite image. Every image trained upon was composited in this way, with the various augmentations randomly generated.

![Figure 4: Augmentation Techniques][image1]

I used all three images per measurement, and offset the left and right images' steering angles by 0.25. I obtained this number from the link above.

I used over 40,000 samples. I then preprocessed this data by smoothing the steering angles and filtering out low-speed samples.

I randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 since beyond that I rarely improved the loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.