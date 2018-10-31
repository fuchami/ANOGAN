# coding:utf-8

import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import os, sys
import math
from tqdm import tqdm

import keras
from keras.datasets import mnist
from keras.optimizers import Adam, RMSprop
from keras.utils.generic_utils import Progbar

import model

def plot_DCGAN_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:4]
    image = np.zeros((height*shape[0], width*shape[1], shape[2]),
                        dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1], :] = img[:, :, :]

    return image

def train_DCGAN(batch_size, X_train):

    """ model load """
    d = model.discriminator_model()
    g = model.generator_model()
    d_on_g = model.generator_containg_discriminator(g, d)

    d_opt = RMSprop(lr=0.0004)
    g_opt = RMSprop(lr=0.002)

    g.compile(loss='mse', optimizer=g_opt)
    d_on_g.compile(loss='mse', optimizer=g_opt)
    d.trainable = True
    d.compile(loss='mse', optimizer=d_opt)

    for epoch in range(10):
        print ("Epoch is ", epoch)
        n_iter = int(X_train.shape[0]/batch_size)
        progress_bar = Progbar(target=n_iter)

        for index in range(n_iter):
            # create random noise -> U(0,1) 10 vactors
            noise = np.random.uniform(0, 1, size=(batch_size, 10))

            """ load real data & generate fake data """
            image_batch = X_train[index*batch_size:(index+1)*batch_size]
            gen_images = g.predict(noise, verbose=0)

            # visualize training result
            if index % 20 == 0:
                image = plot_DCGAN_images(gen_images)
                image =  image*127.5+127.5
                if not os.path.exists('./result/'):
                    os.makedirs('./result/')
                cv2.imwrite('./result/' + str(epoch)+"_"+str(index)+ ".png", image )

            # attach label for training discriminator
            X = np.concatenate((image_batch, gen_images))
            y = np.array([1] * batch_size + [0] * batch_size)

            """ training discriminator """



    return d, g



def run(args):

    """ load mnist data """
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32)) - 127.5 / 127.5
    X_test = (X_test.astype(np.float32)) - 127.5 / 127.5

    X_train = X_train[:,:,:,None]
    X_test = X_test[:,:,:,None]

    X_test_original = X_test.copy()

    X_train = X_train[Y_train==1]
    X_test = X_test[Y_test==1]
    print('train shape: ', X_train.shape)

    """ train DCGAN(generator & discriminator) """
    model_d, model_g = train_DCGAN(args.batchsize, X_train)

    

def main():
    parser = argparse.ArgumentParser(description='train AnoGAN')
    parser.add_argument('--epoch', '-e', default=300)
    parser.add_argument('--batchsize', '-b', default=64)

    args = parser.parse_args()

    run(args)

if __name__ == '__main__':
    main()