#  coding:utf-8

import numpy as np
import cv2
import os, sys
import math
from tqdm import tqdm

import keras
from keras.models import Sequential, Model
from keras.layers import Input, Reshape, Dense, Dropout, MaxPooling2D, Conv2D, Flatten
from keras.layers import Conv2DTranspose, LeakyReLU
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras import initializers
from keras.utils.generic_utils import Progbar
from keras.utils.vis_utils import plot_model

import tensorflow as tf

import model
class DCGAN():
    def __init__(self, args ):

        self.img_size = args.img_size
        self.channels = 1
        self.z_dim = args.zdims
        self.epochs = args.epoch
        self.batch_size = args.batchsize

        self.d_opt = Adam(lr=1e-5, beta_1=0.1)
        self.g_opt = Adam(lr=2e-4, beta_1=0.5)

        if not os.path.exists('./result/'):
            os.makedirs('./result/')
        if not os.path.exists('./model_images/'):
            os.makedirs('./model_images/')

        """ build discriminator model """
        self.d = model.discriminator_model(self.img_size)
        plot_model(self.d, to_file='./model_images/discriminator.png', show_shapes=True)

        """ build generator model """
        self.g = model.generator_model(self.z_dim)
        plot_model(self.g, to_file='./model_images/generator', show_shapes=True)

        """ discriminator on generator model """
        self.d_on_g = model.generator_containg_discriminator(self.g, self.d, self.z_dim)
        plot_model(self.d_on_g, to_file='./model_images/d_on_g', show_shapes=True)

        self.g.compile(loss='mse', optimizer=self.g_opt)
        self.d_on_g.compile(loss='mse', optimizer=self.g_opt)
        self.d.trainable = True
        self.d.compile(loss='mse', optimizer=self.d_opt)

    """ plot images for visualization """
    def plot_generate_images(self, gen_images):
        num = gen_images.shape[0]
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
        shape = gen_images.shape[1:4]
        image = np.zeros((height*shape[0], width*shape[1], shape[2]),
                            dtype=gen_images.dtype)
        for index, img in enumerate(gen_images):
            i = int(index/width)
            j = index % width
            image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1], :] = img[:, :, :]
        return image

    """ generate image """
    def generate(self, batch_size):
        self.g.load_weights('./saved_model/generator.h5')
        noise = np.random.uniform(0, 1, (batch_size, self.z_dim))
        generate_img = self.g.predict(noise)

        return generate_img

    def train(self, X_train):
        for epoch in range(self.epochs):
            print ("Epoch is ", epoch)
            n_iter = int(X_train.shape[0] / self.batch_size)
            progress_bar = Progbar(target=n_iter)

            for index in range(n_iter):
                # create random noise -> U(0,1) 10 vactors
                noise = np.random.uniform(0, 1, size=(self.batch_size, self.z_dim))

                """ load real data & generate fake data """
                image_batch = X_train[index*self.batch_size:(index+1)*self.batch_size]
                gen_images = self.g.predict(noise, verbose=0)

                # visualize training result
                if index % 50 == 0:
                    image = self.plot_generate_images(gen_images)
                    image =  image*127.5+127.5
                    cv2.imwrite('./result/' + str(epoch)+"_"+str(index)+ ".png", image )

                # attach label for training discriminator
                X = np.concatenate((image_batch, gen_images))
                y = np.array([1] * self.batch_size + [0] * self.batch_size)

                """ training discriminator """
                d_loss = self.d.train_on_batch(X, y)

                """ training generator """
                self.d.trainable = False
                g_loss = self.d_on_g.train_on_batch(noise, np.array([1] * self.batch_size))
                self.d.trainable = True

                progress_bar.update(index, values=[('g', g_loss), ('d', d_loss)])
            print('')

            """ save weights for each epoch """
            if not os.path.exists('./saved_model/'):
                os.makedirs('./saved_model/')
            self.g.save_weights('./saved_model/generator.h5', True)
            self.d.save_weights('./saved_model/discriminator.h5',True)

        return self.d, self.g

        

