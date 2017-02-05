import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense
from keras.layers.convolutional import Convolution2D
from keras.models import Model

def process_line(line, path):
    tokens = line.split(',')
    img = mpimg.imread(path + tokens[0])
    angle = float(tokens[3])
    return img, angle

def generate_batch_from_file(path, batch_size, input_shape):
    while 1:
        f = open(path + '/driving_log.csv.relative')
        i = 0
        input_batch = np.empty((batch_size, input_shape[0], input_shape[1], input_shape[2]))
        output_batch = np.empty(batch_size)
        for line in f:
            img, angle = process_line(line, path)
            
            input_batch[i] = (np.asarray(img) - 127.0) / 255.0 # Normalize the image
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
flags.DEFINE_string('training_file', 'training', "Features training file (.csv)")
flags.DEFINE_string('validation_file', 'validation', "Features validation file (.csv)")
flags.DEFINE_integer('epochs', 5, "The number of epochs.")
flags.DEFINE_integer('batch_size', 256, "The batch size.")


def main(_):
    
    # define model
    input_shape = (80,160,3)
    
    print("Input shape: ", input_shape)
    
    model = Sequential()
    
    model.add(Convolution2D(24, 5, 5, border_mode='valid', input_shape=input_shape))
    model.add(Convolution2D(36, 5, 5, border_mode='valid'))
    model.add(Convolution2D(48, 5, 5, border_mode='valid'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    
    model.add(Flatten())
    
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    
    model.fit_generator(generate_batch_from_file(FLAGS.training_file, FLAGS.batch_size, input_shape),
                        samples_per_epoch=FLAGS.batch_size * 10, nb_epoch=FLAGS.epochs)#,
#validation_data=generate_arrays_from_file(FLAGS.validation_file))

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
