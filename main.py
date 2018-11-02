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

""" plot images for visualization """
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

""" generate image """
def generate_img(batch_size):
    g = model.generator_model()
    g.load_weights('./saved_model/generator.h5')
    noise = np.random.uniform(0, 1, (batch_size, 10))
    gen_img = g.predict(noise)

    return generate_img

def train_DCGAN(X_train, batch_size, epoch):

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

    for epoch in range(epoch):
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
            d_loss = d.train_on_batch(X, y)

            """ training generator """
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, np.array([1] * batch_size))
            d.trainable = True

            progress_bar.update(index, values=[('g', g_loss), ('d', d_loss)])
        print('')

        """ save weights for each epoch """
        if not os.path.exists('./saved_model/'):
            os.makedirs('./saved_model/')
        g.save_weights('./saved_model/generator.h5', True)
        d.save_weights('./saved_model/discriminator.h5',True)

    return d, g

def anomaly_detection(test_img, g=None, d=None):
    model = model.anomaly_detector(g=g, d=d)
    ano_score, similar_img = model.compute_anomaly_score(model, test_img.reshape(1, 28, 28, 1), iterations=500, d=d)
    
    # anomaly area, 255 normalization
    np_residual = test_img.reshape(28, 28, 1) - similar_img.reshape(28, 28, 1)
    np_residual = (np_residual +2)/4

    np_residual = (255*np_residual).astype(np.uint8)
    origina_x = (test_img.reshape(28,28,1)*127.5+127.5).astype(np.uint8)
    similar_x = (similar_img.reshape(28,28,1)*127.5+127.5).astype(np.uint8)

    original_x_color = cv2.cvtColor(origina_x, cv2.COLOR_GRAY2BGR)
    residual_color = cv2.applyColorMap(np_residual, cv2.COLORMAP_JET)
    show = cv2.addWighted(original_x_color, 0.3, residual_color, 0.7, 0. )
    
    return ano_score, original_x, similar_x, show

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
    model_d, model_g = train_DCGAN(X_train,args.batchsize, args.epoch)

    """ test generator """
    gen_img = generate_img(25)
    img = plot_DCGAN_images(gen_img)
    img = (img*127.5)+127.5
    img = img.astype(np.uint8)
    img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

    """ openCV view """
    cv2.namedWindow('generated', 0)
    cv2.resizeWindow('generated', 256, 256)
    cv2.imshow('generated', img)
    cv2.imwrite('generator.png', img)
    cv2.waitkey()

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
    score, qurey, pred, diff = anomaly_detection(test_img)
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
    parser.add_argument('--label_idx', type=int, default=7)

    args = parser.parse_args()

    run(args)

if __name__ == '__main__':
    main()