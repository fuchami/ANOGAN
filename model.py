# coding: utf-8

import numpy as np
import cv2
import math

from keras.models import Sequential, Model
from keras.layers import Input, Reshape, Dense, Dropout, MaxPooling2D, Conv2D, Flatten
from keras.layers import Conv2DTranspose, LeakyReLU
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras import initializers
import tensorflow as tf

from keras.utils.generic_utils import Progbar

""" build generator model """
def generator_model(): 
    inputs = Input((10, ))
    fc1 = Dense(input_dim=10, units=128*7*7)(inputs)
    fc1 = BatchNormalization()(fc1)
    fc1 = LeakyReLU(0.2)(fc1)

    fc2 = Reshape((7, 7, 128), input_shape=(128*7*7,))(fc1)
    up1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(fc2)
    conv1 = Conv2D(64, (3,3), padding='same')(up1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    up2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv1)
    conv2 = Conv2D(1, (5, 5), padding='same')(up2)
    outputs = Activation('tanh')(conv2)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary()

    return model
    
""" build discriminator model """
def discriminator_model():
    inputs = Input((28, 28, 1))

    conv1 = Conv2D(64, (5,5), padding='same')(inputs)
    conv1 = LeakyReLU(0.2)(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(128, (5,5), padding='same')(pool1)
    conv2 = LeakyReLU(0.2)(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

    fc1 = Flatten()(pool2)
    fc1 = Dense(1)(fc1)
    outputs = Activation('sigmoid')(fc1)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary()

    return model
    

def generator_containg_discriminator(g, d):
    d.trainable = False

    ganInput = Input(shape=(10, ))
    x = g(ganInput)
    ganOutput = d(x)
    gan = Model(inputs=ganInput, outputs=ganOutput)

    return gan

""" discriminator intermediate ayer feature extraction """
def feature_extractor(d=None):
    if d is None:
        d = discriminator_model()
        d.load_weights('./saved_model/discriminator.h5')
    
    intermidiate_model = Model(inputs=d.layers[0].input, outputs=d.layers[-7].output)
    intermidiate_model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    return intermidiate_model

""" anomaly GAN loss function """
def sum_of_residual(y_true, y_pred):
    return K.sum(K.abs(y_true - y_pred))

""" anomaly detection model """
def anomaly_detector(g=None, d=None):
    if g is None:
        g = generator_model()
        g.loss_weights('./saved_model/generator.h5')
    
    intermidiate_model = feature_extractor(d)
    intermidiate_model.trainable = False

    g = Model(inputs=g.layers[1].input, outputs=g.layers[-1].output)
    g.trainable = False

    # input layer cann't be trained.
    # add new layer as same size & same distribution
    aInput = Input(shape=(10, ))
    gInput = Dense((10), trainable=True)(aInput)
    gInput = Activation('sigmoid')(gInput)

    # g & d feature
    G_out = g(gInput)
    D_out = intermidiate_model(G_out)
    model = Model(inputs=aInput, outputs=[G_out, D_out])
    model.compile(loss=sum_of_residual, loss_weights=[0.90, 0.10], optimizer='rmsprop')

    # batchnorm learning phase fixed (test) : make on trainable
    K.set_learning_phase(0)

    return model

""" anomaly detection """
def compute_anomaly_score(model, x, iterations=500, d=None):
    z = np.random.uniform(0, 1, size=(1, 10))
    
    intermidiate_model = feature_extractor(d)
    d_x = intermidiate_model.predict(x)

    """ learnig for changin latent """
    loss = model.fit(z, [x, d_x], batch_size=1, epochs=iterations, verbose=0)
    similar_data, _ = model.predict(z)

    loss = loss.history['loss'][-1]

    return loss, similar_data



























