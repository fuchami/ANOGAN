# coding:utf-8

import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import os, sys
import math
from tqdm import tqdm

import model
import dcgan

from keras.datasets import mnist

def anomaly_detection(test_img, args, g=None, d=None):
    anogan_model = model.anomaly_detector(args, g=g, d=d)
    ano_score, similar_img = model.compute_anomaly_score(args, anogan_model, test_img.reshape(1, 28, 28, 1), iterations=500, d=d)
    
    # anomaly area, 255 normalization
    np_residual = test_img.reshape(28, 28, 1) - similar_img.reshape(28, 28, 1)
    np_residual = (np_residual + 2)/4

    np_residual = (255*np_residual).astype(np.uint8)
    original_x = (test_img.reshape(28,28,1)*127.5+127.5).astype(np.uint8)
    similar_x = (similar_img.reshape(28,28,1)*127.5+127.5).astype(np.uint8)

    original_x_color = cv2.cvtColor(original_x, cv2.COLOR_GRAY2BGR)
    residual_color = cv2.applyColorMap(np_residual, cv2.COLORMAP_JET)
    show = cv2.addWeighted(original_x_color, 0.3, residual_color, 0.7, 0.)
    
    return ano_score, original_x, similar_x, show

def run(args):

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

    """ init DCGAN """
    print("initialize DCGAN ")
    DCGAN = dcgan.DCGAN(args)

    """ train DCGAN(generator & discriminator) """
    if args.mode == 'train':
        print ('============ train on DCGAN ============')
        DCGAN.train(X_train)
    
    print("trained")

    """ test generator """
    gen_img = DCGAN.generate(25)
    img = DCGAN.plot_generate_images(gen_img)
    img = (img*127.5)+127.5
    img = img.astype(np.uint8)
    img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

    """ openCV view """
    #cv2.namedWindow('generated', 0)
    #cv2.resizeWindow('generated', 256, 256)
    #cv2.imshow('generated', img)
    #cv2.imwrite('generator.png', img)
    #cv2.waitKey()

    """ plt view """
    plt.figure(num=0, figsize=(4, 4))
    plt.title('trained generator')
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()

    """ other class anomaly detection """
    # compute anomaly score - sample from test set
    #test_img = X_test_original[Y_test==1][30]
    # compute anomaly score - sample from strange image
    #test_img = X_test_original[Y_test==0][30]

    # compute anomaly score - sample from strange image
    img_idx = args.img_idx
    label_idx = args.label_idx
    test_img = X_test_original[Y_test==label_idx][img_idx]
    # test_img = np.random.uniform(-1, 1 (28, 28, 1))

    start = cv2.getTickCount()
    score, qurey, pred, diff = anomaly_detection(test_img, args)
    time = (cv2.getTickCount() - start ) / cv2.getTickFrequency() * 1000
    print ('%d label, %d : done ' %(label_idx, img_idx), '%.2f' %score, '%.2fms'%time)

    """ matplot view """
    plt.figure(1, figsize=(3, 3))
    plt.title('query image')
    plt.imshow(qurey.reshape(28, 28), cmap=plt.cm.gray)

    print('anomaly score :', score)
    plt.figure(2, figsize=(3,3))
    plt.title('generated similar image')
    plt.imshow(pred.reshape(28, 28), cmap=plt.cm.gray)

    plt.figure(3, figsize=(3,3))
    plt.title('anomaly detection')
    plt.imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='train AnoGAN')
    parser.add_argument('--epoch', '-e', default=30)
    parser.add_argument('--batchsize', '-b', default=64)
    parser.add_argument('--img_idx', type=int, default=14)
    parser.add_argument('--label_idx', type=int, default=0)
    parser.add_argument('--mode', '-m' , type=str, default='test',help='train, test')
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--zdims', type=int, default=10)

    args = parser.parse_args()

    run(args)

if __name__ == '__main__':
    main()