import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import cv2
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Model
from keras.optimizers import Adam

def process_line(line, path):
    tokens = line.split(',')
    center = mpimg.imread(path + tokens[0].strip())
    left = mpimg.imread(path + tokens[1].strip())
    right = mpimg.imread(path + tokens[2].strip())
    center = cv2.resize(center, (64,64))
    left = cv2.resize(left, (64,64))
    right = cv2.resize(right, (64,64))
    img = (np.concatenate((left, center, right), axis = 1) - 127.0) / 255.0 # Normalize the image
    angle = float(tokens[3].strip())
    return img, angle

def generate_batch_from_file(path, batch_size, input_shape):
    while 1:
        f = open(path + '/driving_log.csv')
        i = 0
        input_batch = np.empty((batch_size, input_shape[0], input_shape[1], input_shape[2]))
        output_batch = np.empty((batch_size))
        for line in f:
            img, angle = process_line(line, path)
            
            input_batch[i] = img
            output_batch[i] = angle
    
            if (i == batch_size-1):
                yield (input_batch, output_batch)
                i = 0
            else:
                i += 1
    
        f.close()

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', 'training/', "Features training file (.csv)")
flags.DEFINE_string('validation_file', 'validation/', "Features validation file (.csv)")
flags.DEFINE_integer('epochs', 5, "The number of epochs.")
flags.DEFINE_integer('batch_size', 128, "The batch size.")
flags.DEFINE_float('learning_rate', 0.001, "The learning rate.")
flags.DEFINE_integer('batch_multiple', 30, "The batch multiple.")

def main(_):
    
    # define model
    input_shape = (64,192,3)
    
    print("Input shape: ", input_shape)
    
    model = Sequential()
    
    model.add(Convolution2D(24, 5, 5, border_mode='valid', input_shape=input_shape))
    model.add(Convolution2D(36, 5, 5, border_mode='valid'))
    model.add(Convolution2D(48, 5, 5, border_mode='valid'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))

    model.add(Flatten())

    model.add(Dropout(0.5))

    model.add(Dense(100, activation='relu'))

    model.add(Dropout(0.8))

    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    adam = Adam(lr=FLAGS.learning_rate)
    model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
    
    model.fit_generator(generate_batch_from_file(FLAGS.training_file, FLAGS.batch_size, input_shape),
                        samples_per_epoch=FLAGS.batch_size * FLAGS.batch_multiple, nb_epoch=FLAGS.epochs)#,
#validation_data=generate_arrays_from_file(FLAGS.validation_file))

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
