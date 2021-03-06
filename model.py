import tensorflow as tf
import random, copy
from os import path
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam

"""
Attribution:

I took inspiration from https://github.com/dolaameng/Udacity-SDC_Behavior-Cloning/blob/master/sdc/data.py when constructing my preprocessing 
(DataSet) class, although I modified this class heavily. This was my first experience with pandas, which greatly simplified the parsing of
CSV data, which I was doing manually before taking a cue from dolaameng's implementation.

For the generator class, I used the approach shown in https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.p5th2guac.
I improved upon the shadow casting method by using the HSV rather than HLS color space, as the HLS color space was producing strange artifacts.
Furthermore, I use every image generated by the simulator (center, left, and right) rather than randomly selecting one per sample.

"""

"""
This class represents all of the data in the training and cross-validation sets. It accepts a comma-delimited list of folder paths, each
of which is expected to contain a CSV file called "driving_log.csv" and a folder called "IMG" containing images referred to in the driving log.
The driving log contains three image references per sample, corresponding to the center, left, and right dashboard camera images captured
by the simulator. Each sample also contains numerical metrics, of which I used Speed during preprocessing. The SteeringAngle metric is the
output variable that should be predicted by the camera images.

The list of folders passed to the constructor makes adding and removing bits of training data easy. If a training run was poor and hurt performance,
that folder could be excluded from further training. Likewise if more training was needed, additional training data could be added incrementally.

All training data is merged, shuffled, and split into a training set (80%) and a cross-validaiton set (20%). In addition before merging, the
steering angles are passed through a low-pass filter to smooth out the sometimes jerky measurements from the control devices. I found empirically
that a window of 4 samples gave the best smoothing results.

After merging, all low-speed measurements are filtered out, to avoid capturing times when the car is idling, stuck, or otherwise engaging in 
behavior unlikely to be replicated in a testing run. (fingers crossed ;)

The preprocess() method returns two generators, one to be used for training and the other for cross-validation.
"""
class DataSet(object):
    _headers = ["CenterImage", "LeftImage", "RightImage", "SteeringAngle", "Throttle", "Brake", "Speed"]
    _steering_angle_smoothing_window = 4
    _minimum_speed = 20.0
    _steering_angle_augmentation = 0.25
    _training_fraction = 0.8
    
    def __init__(self, data_paths):
        # A data path is the folder containing a file called "driving_log.csv"
        # and a folder called "IMG" containing the images referred to in that
        # log file
        logs = []
        
        for data_path in data_paths:
            image_folder = data_path + "/IMG"
            log = pd.read_csv(data_path + "/driving_log.csv", header=None, names=self._headers)
            for column in ["CenterImage", "LeftImage", "RightImage"]:
                log[column] = log[column].str.rsplit("/", n=1).str[-1].apply(lambda p: path.join(image_folder, p))
            log = self._smooth_steering_angle(log)
            logs.append(log)
            
        self.log = pd.concat(logs, axis=0, ignore_index=True)

    def _smooth_steering_angle(self, log):
        log.SteeringAngle = log.SteeringAngle.rolling(window=self._steering_angle_smoothing_window, center=True).mean()
        log.SteeringAngle = log.SteeringAngle.fillna(0)
        return log
    
    def preprocess(self):
        self._remove_low_speeds()
        self._flatten()
        self._shuffle()
        self._validate()
        self._split()
        return (CloningGenerator(self.training_keys, self.samples), CloningGenerator(self.validation_keys, self.samples))
    
    def _remove_low_speeds(self):
        self.log = self.log[self.log.Speed >= self._minimum_speed]
        return self

    """
    Only use image measurements whose image files exist.
    """
    def _validate(self):
        self.sample_keys = [key for key in self.sample_keys if path.isfile(key)]
        return self

    """
    The steering angle augmentation value was taken from https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.p5th2guac.
    """
    def _flatten(self):
        self.samples = dict(zip(self.log.CenterImage, self.log.SteeringAngle))
        self.samples.update(zip(self.log.LeftImage, self.log.SteeringAngle + self._steering_angle_augmentation))
        self.samples.update(zip(self.log.RightImage, self.log.SteeringAngle - self._steering_angle_augmentation))
        return self
    
    def _shuffle(self):
        self.sample_keys = list(self.samples.keys())
        random.shuffle(self.sample_keys)
        return self

    def _split(self):
        self.training_keys, self.validation_keys = train_test_split(self.sample_keys, train_size=self._training_fraction)
        return self

"""

This generator class creates training batches in a separate thread, to avoid loading all training images into memory.

This class will augment images taken from disk by:

    * adjusting the image brightness, 
    * changing the image position along with the steering angle, 
    * casting random shadows, 
    * cropping the sky and dashboard from the image, 
    * scaling the image to 64x64 pixels to improve memory use during training,
    * normalizing image depth to [-05,0.5], and
    * randomly flipping the image horizontally, and flipping the sign on the steering angle to match.
    
This augmentation enabled my model to pass the second course on the first try! (I am assuming reaching the insurmountable
slope is as far as we're supposed to go. I tried increasing the throttle to no effect.)

"""
class CloningGenerator(object):
    _trans_range = 100
    
    def __init__(self, sample_keys, samples):
        self._sample_keys = sample_keys
        self._samples = samples
        
    def augment_brightness(self, image):
        bright = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        random_bright = .5 + 0.5 * np.random.uniform()
        bright[:,:,2] = bright[:,:,2] * random_bright
        bright = cv2.cvtColor(bright, cv2.COLOR_HSV2RGB)
        return bright

    def translate_image(self, image, angle):
        height, width, depth = image.shape
        tr_x = self._trans_range * np.random.uniform() - self._trans_range / 2
        steer_angle = angle + tr_x / self._trans_range * 2 * .2
        tr_y = 5 * np.random.uniform() - 5 / 2
        Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
        image_tr = cv2.warpAffine(image, Trans_M, (width, height))
        return image_tr, steer_angle

    def add_random_shadow(self, image):
        height, width, depth = image.shape
        top_y = height * np.random.uniform()
        top_x = 0
        bot_x = width
        bot_y = height * np.random.uniform()
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        shadow_mask = 0 * image_hsv[:,:,2]
        X_m = np.mgrid[0:height, 0:width][0]
        Y_m = np.mgrid[0:height, 0:width][1]

        shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >= 0)] = 1
        
        if np.random.randint(2) == 1:
            bright = .5
            cond1 = shadow_mask == 1
            cond0 = shadow_mask == 0
            if np.random.randint(2) == 1:
                image_hsv[:,:,2][cond1] = image_hsv[:,:,2][cond1] * bright
            else:
                image_hsv[:,:,2][cond0] = image_hsv[:,:,2][cond0] * bright    
        image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)

        return image

    def crop_image(self, image):
        height, width, depth = image.shape
        # Remove the sky
        bottom = int(height / 4.0)
        # Remove the hood and dashboard
        top = int(height / 8.0) * 7
        return image[bottom:top, 0:width]
    
    def scale_image(self, image):
        return cv2.resize(image, (64,64))

    def normalize_image(self, image):
        return image / 255. - 0.5

    def mirror_image(self, image, steering_angle):
        if np.random.randint(2) == 0:
            return cv2.flip(image,1), steering_angle * -1
        return image, steering_angle

    def transform_sample(self, image_path, steering_angle):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.augment_brightness(image)
        image, steering_angle = self.translate_image(image, steering_angle)
        image = self.add_random_shadow(image)
        image = self.crop_image(image)
        image = self.scale_image(image)
        image, steering_angle = self.mirror_image(image, steering_angle)
        image = self.normalize_image(image)
        return image, steering_angle
        
    def generate_batches(self, batch_size):
        input_batch = np.empty((batch_size, 64, 64, 3))
        output_batch = np.empty((batch_size))
        batch_count = int(len(self._sample_keys) / batch_size)

        while True:
            for batch in range(batch_count):
                offset = batch*batch_size
                for sample_index in range(batch_size):
                    key = self._sample_keys[offset + sample_index]

                    image, steering_angle = self.transform_sample(key, self._samples[key])
                    
                    input_batch[sample_index] = image
                    output_batch[sample_index] = steering_angle

                yield (input_batch, output_batch)

    def get_set_size(self, batch_size):
        return int(len(self._sample_keys) / batch_size) * batch_size


flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('sources', "data", "Folders to get data from.")
flags.DEFINE_integer('epochs', 1, "The number of epochs.")
flags.DEFINE_integer('batch_size', 64, "The batch size.")
flags.DEFINE_float('learning_rate', 0.001, "The learning rate.")

def main(_):
    
    model = Sequential()
    
    """
    
    This model was inspired by https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.p5th2guac. 
    
    The convolutional layers progressively flatten visual features, and drop 50% of them each layer to prevent overfitting.
    
    """
    
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(64,64,3)))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(128, 3, 3, border_mode='valid'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(16, activation='relu'))
    
    """
    A final layer with 1 output is used with a linear activation since we are outputting a single, continuous value.
    """
    model.add(Dense(1, activation='linear'))
    
    """
    Although the learning rate is a tunable hyperparameter, I left it as the default throughout training.
    """
    adam = Adam(lr=FLAGS.learning_rate)
    
    """
    A mean squared error loss function was used since this is linear rather than logistic regression.
    """
    model.compile(optimizer=adam, loss='mse')

    data_set = DataSet(FLAGS.sources.split(','))
    training_set, validation_set = data_set.preprocess()
    
    training_set_size = training_set.get_set_size(FLAGS.batch_size)
    print("Training set size: " + str(training_set_size))
    validation_set_size = validation_set.get_set_size(FLAGS.batch_size)
    print("Validation set size: " + str(validation_set_size))
    
    model.fit_generator(training_set.generate_batches(FLAGS.batch_size),
                        samples_per_epoch=training_set_size, nb_epoch=FLAGS.epochs,
                        validation_data=validation_set.generate_batches(FLAGS.batch_size),
                        nb_val_samples=validation_set_size)

    model.save('model.h5')

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
