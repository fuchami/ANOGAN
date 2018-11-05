# coding:utf-8

import numpy as np
import os, sys
import glob
import csv

from keras.preprocessing.image import load_img, img_to_array
from keras.datasets import mnist
from sklearn.model_selection import train_test_split

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

def load_csv_data(dataset_path,img_size):
    X_ = []
    X_data = []
    Y_data = []

    print(dataset_path)
    with open(dataset_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            Y_data.append(int(row[-1]))
            X_ = list(map(int, row[:img_size*img_size]))
            X_ = np.array(X_).reshape(img_size, img_size)
            
            #print('X_.shape:', X_.shape)
            X_data.append(X_)

        train_len = int(len(X_)*0.9)
        validation_len = len(X_) - train_len
        X_train, X_test, Y_train, Y_test =\
            train_test_split(X_data, Y_data, test_size=validation_len)

        X_train = (np.array(X_train).astype(np.float32) - 127.5) / 127.5
        X_test = (np.array(X_test).astype(np.float32) - 127.5) / 127.5
        Y_train = np.array(Y_train)
        Y_test = np.array(Y_test)

        X_train = X_train[:,:,:,None]
        X_test = X_test[:,:,:,None]

        X_test_original = X_test.copy()

        print('Y_data.shape:', Y_train.shape)
        print('x_data.shape', X_train.shape)

        X_train = X_train[Y_train==1]
        X_test = X_test[Y_test==1]
        

        return X_train, X_test, X_test_original, Y_test 

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