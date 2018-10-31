# coding:utf-8

import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import os, sys
from keras.datasets import mnist

import model

def run(args):

    """ load mnist data """
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = (X_train)

def main():
    parser = argparse.ArgumentParser(description='train AnoGAN')
    parser.add_argument('--epoch', '-e', default=300)

    args = parser.parse_args()

    run(args)

if __name__ == '__main__':
    main()