import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import cv2
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Model
from keras.optimizers import Adam

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def translate_image(image,steer,trans_range, y_dim, x_dim):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(x_dim,y_dim))
    return image_tr,steer_ang

def add_random_shadow(image, y_dim, x_dim):
    top_y = y_dim*np.random.uniform()
    top_x = 0
    bot_x = x_dim
    bot_y = y_dim*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]

    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)

    return image

def process_line(line, path, mini_batch_size, y_dim, x_dim, z_dim, angle_adjust_left = 0.25, angle_adjust_right = -0.25):
    tokens = line.split(',')
    center = mpimg.imread(path + tokens[0].strip())
    left = mpimg.imread(path + tokens[1].strip())
    right = mpimg.imread(path + tokens[2].strip())

    #Resize the images
    center = cv2.resize(center, (y_dim,x_dim))
    left = cv2.resize(left, (y_dim,x_dim))
    right = cv2.resize(right, (y_dim,x_dim))

    angle = float(tokens[3].strip())

    images = np.empty((mini_batch_size, y_dim, x_dim, z_dim), dtype=np.float32)
    angles = np.empty((mini_batch_size), dtype=np.float32)

    images[0] = center
    images[1] = left
    images[2] = right

    angles[0] = angle
    angles[1] = angle + angle_adjust_left
    angles[2] = angle + angle_adjust_right

 #   images[3], angles[3] = translate_image(images[0].copy(),angles[0],100, y_dim, x_dim)
#    images[4], angles[4] = translate_image(images[1].copy(),angles[1],100, y_dim, x_dim)
#    images[5], angles[5] = translate_image(images[2].copy(),angles[2],100, y_dim, x_dim)

    #images[6] = add_random_shadow(images[0].copy(), y_dim, x_dim)
    images[3] = augment_brightness_camera_images(images[0].copy())

 #   angles[6] = angle
    angles[3] = angle

    for i in range(mini_batch_size):
        # Normalize the images
        images[i] = (images[i] - 127.0) / 255.0
        ind_flip = np.random.randint(2)
        if ind_flip==0:
            images[i] = cv2.flip(images[i],1)
            angles[i] = -angles[i]
    
    return images, angles

def generate_batch_from_file(path, file_name, batch_size, mini_batch_size, input_shape):
    while 1:
        f = open(path + file_name)
        i = 0
        input_batch = np.empty((batch_size, input_shape[0], input_shape[1], input_shape[2]))
        output_batch = np.empty((batch_size))
        for line in f:
            images, angles = process_line(line, path, mini_batch_size, input_shape[0], input_shape[1], input_shape[2])
            
            input_batch[i:i+mini_batch_size] = images
            output_batch[i:i+mini_batch_size] = angles
    
            if (i >= batch_size-mini_batch_size):
                yield (input_batch, output_batch)
                i = 0
            else:
                i += mini_batch_size

        f.close()

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_path', 'training/', "Features training path")
flags.DEFINE_string('training_file', 'driving_log.training.csv', "Features training file (.csv)")
flags.DEFINE_string('validation_file', 'driving_log.validation.csv', "Features validation file (.csv)")
flags.DEFINE_integer('epochs', 5, "The number of epochs.")
flags.DEFINE_integer('batch_size', 64, "The batch size.")
flags.DEFINE_integer('mini_batch_size', 4, "The mini batch size.")
flags.DEFINE_float('learning_rate', 0.001, "The learning rate.")
flags.DEFINE_integer('batch_multiple', 60, "The batch multiple.")

def main(_):
    
    # define model
    input_shape = (64,64,3)
    
    model = Sequential()
    
    model.add(Convolution2D(8, 4, 4, border_mode='valid', input_shape=input_shape))
    model.add(Convolution2D(16, 4, 4, border_mode='valid'))

    model.add(Flatten())

    model.add(Dropout(0.5))

    model.add(Dense(100, activation='relu'))

    model.add(Dropout(0.8))

    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    adam = Adam(lr=FLAGS.learning_rate)
    model.compile(optimizer=adam, loss='mse', metrics = ['accuracy'])

    training_samples = FLAGS.batch_size * FLAGS.mini_batch_size * FLAGS.batch_multiple
    
    model.fit_generator(generate_batch_from_file(FLAGS.training_path, FLAGS.training_file, FLAGS.batch_size, FLAGS.mini_batch_size, input_shape),
                        samples_per_epoch=training_samples, nb_epoch=FLAGS.epochs, nb_val_samples=training_samples/5,
                        validation_data=generate_batch_from_file(FLAGS.training_path, FLAGS.validation_file, FLAGS.batch_size, FLAGS.mini_batch_size, input_shape))

    model.save('model.h5')

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
