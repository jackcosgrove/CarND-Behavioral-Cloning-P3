import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import cv2
import math
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

def translate_image(image,steer,trans_range):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,image.shape[0:2])
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

new_size_col,new_size_row = 64, 64

def preprocessImage(image):
    shape = image.shape
    # note: numpy arrays are (row, col)!
    image = image[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(new_size_col,new_size_row), interpolation=cv2.INTER_AREA)    
    image = image/255.-.5
    return image

def preprocess_image_file_train(line_data, path):
    tokens = line_data.split(',')
    i_lrc = np.random.randint(3)
    if (i_lrc == 0):
        path += tokens[1].strip()
        shift_ang = .25
    if (i_lrc == 1):
        path += tokens[0].strip()
        shift_ang = 0.
    if (i_lrc == 2):
        path += tokens[2].strip()
        shift_ang = -.25
    y_steer = float(tokens[3]) + shift_ang
    image = mpimg.imread(path)
    #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image,y_steer = translate_image(image,y_steer,100)
    image = augment_brightness_camera_images(image)
    image = preprocessImage(image)
    image = np.array(image)
    ind_flip = np.random.randint(2)
    if ind_flip==0:
        image = cv2.flip(image,1)
        y_steer = -y_steer

    return image,y_steer

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

#    images[3], angles[3] = translate_image(images[0].copy(),angles[0],100, y_dim, x_dim)
#    images[4], angles[4] = translate_image(images[1].copy(),angles[1],100, y_dim, x_dim)
#    images[5], angles[5] = translate_image(images[2].copy(),angles[2],100, y_dim, x_dim)

    images[3] = add_random_shadow(images[0].copy(), y_dim, x_dim)
#    images[7] = add_random_shadow(images[0].copy(), y_dim, x_dim)

    angles[3] = angle
#    angles[7] = angle

    for i in range(mini_batch_size):
        # Normalize the images
        images[i] = (images[i] - 127.0) / 255.0
        ind_flip = np.random.randint(2)
        if ind_flip==0:
            images[i] = cv2.flip(images[i],1)
            angles[i] = -angles[i]
    
    return images, angles

def generate_batch_from_file(path, data, batch_size, input_shape):
    input_batch = np.empty((batch_size, new_size_row, new_size_col, 3))
    output_batch = np.empty((batch_size))
    while 1:
        num_lines = len(data)
        pr_threshold = 0.5
        for i_batch in range(batch_size):
            i_line = np.random.randint(num_lines)
            line_data = data[i_line]

            keep_pr = 0
            while keep_pr == 0:
                x,y = preprocess_image_file_train(line_data, path)
                pr_unif = np.random
                if abs(y)<.1:
                    pr_val = np.random.uniform()
                    if pr_val>pr_threshold:
                        keep_pr = 1
                else:
                    keep_pr = 1
            
            input_batch[i_batch] = x
            output_batch[i_batch] = y
        yield input_batch, output_batch

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_path', 'training/', "Features training path")
flags.DEFINE_string('training_file', 'driving_log.csv.training', "Features training file (.csv)")
flags.DEFINE_string('validation_file', 'driving_log.csv.validation', "Features validation file (.csv)")
flags.DEFINE_integer('epochs', 5, "The number of epochs.")
flags.DEFINE_integer('batch_size', 64, "The batch size.")
flags.DEFINE_float('learning_rate', 0.001, "The learning rate.")
flags.DEFINE_integer('batch_multiple', 60, "The batch multiple.")

def main(_):
    
    # define model
    input_shape = (160,320,3)
    
    model = Sequential()
    
    model.add(Convolution2D(24, 5, 5, border_mode='valid', input_shape=(new_size_col,new_size_row,3)))
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

    training_samples_count = FLAGS.batch_size * FLAGS.batch_multiple

    f = open(FLAGS.training_path + FLAGS.training_file)
    training_samples = f.readlines()
    f.close()

    f = open(FLAGS.training_path + FLAGS.validation_file)
    validation_samples = f.readlines()
    f.close()
    
    model.fit_generator(generate_batch_from_file(FLAGS.training_path, training_samples, FLAGS.batch_size, input_shape),
                        samples_per_epoch=training_samples_count, nb_epoch=FLAGS.epochs, nb_val_samples=training_samples_count/5,
                        validation_data=generate_batch_from_file(FLAGS.training_path, validation_samples, FLAGS.batch_size, input_shape))

    model.save('model.h5')

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
