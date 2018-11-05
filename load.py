# coding:utf-8

import numpy as np
import os, sys
import glob

from keras.preprocessing.image import load_img, img_to_array
from keras.datasets import mnist

def load_image_data(dataset_path, test_img, img_size, mode):
    X_train = []
    X_test = []

    print(dataset_path)
    print("train image data loading...")
    train_image_list = glob.glob(dataset_path+'*.jpg')
    for img_path in train_image_list:
        print(img_path)
        img = load_img(img_path, target_size=(img_size, img_size))
        imgarray = img_to_array(img)
        X_train.append(imgarray)
    
    X_train = np.array(X_train).astype(np.float32)
    X_train = (X_train -127.5) / 127.5
    
    if mode == 'test':
        test_img = load_img(test_img, target_size=(img_size, img_size))
        test_imgarray = img_to_array(test_img)
        X_test = np.array(test_imgarray).astype(np.float32)
        X_test = (X_test -127.5) / 127.5

    return X_train, X_test

def load_mnist_data():
    """ load mnist data """
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_test = (X_test.astype(np.float32) - 127.5) / 127.5

    X_train = X_train[:,:,:,None]
    X_test = X_test[:,:,:,None]

    X_test_original = X_test.copy()

    X_train = X_train[Y_train==1]
    X_test = X_test[Y_test==1]
    print('train shape: ', X_train.shape)

    return X_train, X_test, X_test_original, Y_test 