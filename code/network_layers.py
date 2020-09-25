import numpy as np
import scipy.ndimage
import skimage
import os

def extract_deep_feature(x, vgg16_weights):
    '''
    Extracts deep features from the given VGG-16 weights.

    [input]
    * x: numpy.ndarray of shape (H, W, 3)
    * vgg16_weights: list of shape (L, 3)

    [output]
    * feat: numpy.ndarray of shape (K)
    '''

    # resize image to 224x224 and normalize
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    x = skimage.transform.resize(x, (224,224))
    x = (x-mean)/std

    # loop through layers
    linear_count = 0
    for layer in vgg16_weights:
        if linear_count >= 2:
            break
        layer_str = layer[0]
        W = layer[1]
        b = layer[2]
        if 'conv2d' == layer_str:
            x = multichannel_conv2d(x, W, b)
        elif 'relu' == layer_str:
            x = relu(x)
        elif 'maxpool2d' == layer_str:
            size = layer.kernel_size
            x = max_pool2d(x, size)
        elif 'linear' == layer_str:
            x = linear(x,W,b)
            linear_count += 1
    return x

def multichannel_conv2d(x, weight, bias):
    '''
    Performs multi-channel 2D convolution.

    [input]
    * x: numpy.ndarray of shape (H, W, input_dim)
    * weight: numpy.ndarray of shape (output_dim, input_dim, kernel_size, kernel_size)
    * bias: numpy.ndarray of shape (output_dim)

    [output]
    * feat: numpy.ndarray of shape (H, W, output_dim)
    '''

    # TODO

    m,n,depth = x.shape

    # for each filter

        # for each channel

            # x[:,:,channel] = scipy.ndimage.convolve(x[:,:,channel],filter,b)
    return x

def relu(x):
    '''
    Rectified linear unit.

    [input]
    * x: numpy.ndarray

    [output]
    * y: numpy.ndarray
    '''

    zeros = np.zeros((x.shape[0],))
    return np.maximum(x,zeros)

def max_pool2d(x, size):
    '''
    2D max pooling operation.

    [input]
    * x: numpy.ndarray of shape (H, W, input_dim)
    * size: pooling receptive field

    [output]
    * y: numpy.ndarray of shape (H/size, W/size, input_dim)
    '''

    m,n,depth = x.shape

    # pad x to make m,n divisible by size
    pad_height = int(np.ceil(m/size)*size-m)
    pad_width = int(np.ceil(n/size)*size-n)
    x = np.pad(x, ((pad_height,0),(pad_width,0),(0,0)), 'minimum')
    m,n,depth = x.shape

    # https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy
    x_new = np.zeros((int(m/size),int(n/size),depth))
    # for each channel, do
    for i in range(depth):
        x_new[:,:,i] = x[:,:,i].reshape(int(m/size),size,int(n/size),size).max(axis=(1,3))

    return x_new

def linear(x,W,b):
    '''
    Fully-connected layer.

    [input]
    * x: numpy.ndarray of shape (input_dim)
    * weight: numpy.ndarray of shape (output_dim,input_dim)
    * bias: numpy.ndarray of shape (output_dim)

    [output]
    * y: numpy.ndarray of shape (output_dim)
    '''

    y = np.matmul(W,x)+b
    return y
