import numpy as np
import cv2
import cube_data
import math
import cv2
import random
import keras
from keras.layers import Input, Merge, MaxPooling2D, LSTM, BatchNormalization, AtrousConv2D
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.optimizers import Adam,SGD,RMSprop,Adadelta
from keras import initializers
from keras.models import load_model
import h5py

def make_generator(my_image_width):
    loc_rot_and_light = Input(shape=(4,))

    hidden1 =  Dense(30, activation='relu')(loc_rot_and_light)
    hidden2 = Dense(150, activation='relu')(hidden1)
    bn = BatchNormalization()(hidden2)
    square = Dense(my_image_width * my_image_width, activation='sigmoid')(bn)
    reshaped = Reshape((my_image_width, my_image_width))(square)

    model = Model(inputs=loc_rot_and_light, outputs=reshaped)
    # print(model.summary())
    return model

def write_prediction(my_gen, my_super_epoch, my_output_directory):
    sampled_x = np.random.uniform(0, 1.0, 4)
    sampled_x = np.reshape(sampled_x, (1,4))
    pred = my_gen.predict(sampled_x)
    im = pred[0]
    im = im *128.0
    filename = output_directory + 'prediction_' + str(my_super_epoch) + '.png'
    cv2.imwrite(filename, im)

# ----------------------------------------------------------------
output_directory = '/home/foo/data/blend-new/dataset_cube/output/'

image_width = 32
x, y = cube_data.get_all_data('/home/foo/data/blend-new/dataset_cube/images', image_width)
gen = make_generator(image_width)

opt = SGD()
gen.compile(loss='mean_squared_error', optimizer=opt)

for super_epoch in range(0,1000):
    gen.fit(x, y, epochs=100, batch_size=16, verbose=1, validation_split=.15)

    write_prediction(gen, super_epoch, output_directory)







