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
    reshaped = Reshape((my_image_width, my_image_width))(square) #?1

    model = Model(inputs = loc_rot_and_light, outputs = reshaped)
    print(model.summary())
    return model

#----------------------------------------------------------------
image_width = 32
x, y = cube_data.get_all_data('/home/foo/data/blend-new/dataset_cube/images', image_width)
gen = make_generator(image_width)

opt = SGD()
gen.compile(loss='mean_squared_error', optimizer=opt)
gen.fit(x, y, epochs=10000, batch_size=16, verbose=1, validation_split=.15)






