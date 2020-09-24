import numpy as np
from multiprocessing import Pool
import scipy.ndimage
import skimage
import sklearn.cluster
import scipy.spatial.distance
import os, time
import matplotlib.pyplot as plt
import util
import random
import cv2

def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)

    [output]
    * filter_responses: numpy.ndarray of shape (H, W, 3F)
    '''

    filters = ['Gaussian', 'Laplacian', 'DoGx', 'DoGy']
    scales = [1., 2., 4., 8., 8*2**(0.5)]
    F = len(filters)*len(scales)

    # if grayscale, convert to 3 channels
    grayscale = len(image.shape)==2
    if grayscale:
        # print('Converting image to 3 channels')
        image = np.stack((image,)*3, axis=-1)

    # check data type
    if image.dtype != 'float':
        # print('Converting image from', image.dtype,'to float')
        image = image.astype('float')

    # check range of values
    if not ((image >= 0.).all() and (image <= 1.).all()):
        # print('Incorrect range')
        image = image/255.

    image = skimage.color.rgb2lab(image)
    # plt.imshow(image);plt.show()

    m,n,channels = image.shape
    results = np.zeros((m,n,3*F))
    f = 0
    for scale in scales:
        for filter in filters:
            if filter == 'Gaussian':
                results[:,:,3*f:3*f+3] = scipy.ndimage.gaussian_filter(image, sigma=scale, mode='nearest')
            elif filter == 'Laplacian':
                results[:,:,3*f:3*f+3] = scipy.ndimage.gaussian_laplace(image, sigma=scale, mode='nearest')
            elif filter == 'DoGx':
                results[:,:,3*f:3*f+3] = scipy.ndimage.gaussian_filter(image, sigma=scale, order=(0,1,0), mode='nearest')
            elif filter == 'DoGy':
                results[:,:,3*f:3*f+3] = scipy.ndimage.gaussian_filter(image, sigma=scale, order=(1,0,0), mode='nearest')
            # plt.imshow(results[:,:,3*f:3*f+3]);plt.show()
            f += 1
    return results

def get_visual_words(image, dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)

    [output]
    * wordmap: numpy.ndarray of shape (H, W)
    '''

    # ----- TODO -----
    # if grayscale, convert to 3 channels
    grayscale = len(image.shape)==2
    if grayscale:
        # print('Converting image to 3 channels')
        image = np.stack((image,)*3, axis=-1)

    rows,cols,depth = image.shape
    k,word_length = dictionary.shape
    filter_images = extract_filter_responses(image)

    # want to compare H*W x 3F with k x 3F
    filter_images = filter_images.reshape((rows*cols,word_length))
    dists = scipy.spatial.distance.cdist(filter_images, dictionary, 'euclidean')

    wordmap = np.argmin(dists, axis=1).reshape((rows,cols))
    return wordmap.astype('int32')

def get_corners(Rs, alpha):
    corner_pixels = np.zeros((alpha,2))
    corner_values = np.zeros((alpha,))
    rows, cols = Rs.shape
    Rs = Rs.flatten()
    for i in range(rows*cols):
        min_index = np.argmin(corner_values)
        if Rs[i] > corner_values[min_index]:
            m = int(i/cols)
            n = int(i%cols)
            corner_pixels[min_index] = (m,n)
            corner_values[min_index] = Rs[i]
    return corner_pixels.astype('int32')

def get_harris_points(image, alpha, k=0.05):
    '''
    Compute points of interest using the Harris corner detector

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)
    * alpha: number of points of interest desired
    * k: senstivity factor

    [output]
    * points_of_interest: numpy.ndarray of shape (alpha, 2) that contains interest points
    '''

    # check data type
    if image.dtype != 'float32':
        # print('Converting image from', image.dtype,'to float32')
        image = image.astype('float32')

    # convert to grayscale
    if len(image.shape)==3:
        # print('Converting image to grayscale')
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    rows,cols = image.shape
    Rs = np.zeros(image.shape)

    sobel_x = scipy.ndimage.sobel(image, axis=1, mode='nearest')
    sobel_y = scipy.ndimage.sobel(image, axis=0, mode='nearest')
    for m in range(rows):
        for n in range(cols):
            # define image patch and image gradients
            Ix = sobel_x[m:m+3,n:n+3]
            Iy = sobel_y[m:m+3,n:n+3]

            # build covariance matrix
            Ixx = np.sum(np.multiply(Ix,Ix))
            Ixy = np.sum(np.multiply(Ix,Iy))
            Iyy = np.sum(np.multiply(Iy,Iy))

            # calculate R
            det = Ixx*Iyy - Ixy*Ixy
            tr = Ixx + Iyy
            Rs[m,n] = det - k*tr*tr
    return get_corners(Rs, alpha)

def compute_dictionary_one_image(args):
    '''
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of random samples
    * image_path: path of image file

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha, 3F)
    '''


    i, alpha, image_path = args
    print('Computing dictionary for image',i)
    image = skimage.io.imread(image_path)
    image = image.astype('float')/255
    # plt.imshow(image);plt.show()

    corners = get_harris_points(image,alpha)
    filter_images = extract_filter_responses(image)
    responses = np.zeros((alpha,filter_images.shape[2]))
    for k in range(0,filter_images.shape[2],3):
        filter_image = filter_images[:,:,k:k+3]
        responses[:,k:k+3] = filter_image[corners[:,0],corners[:,1]]

    # save filter responses to a temp file
    filename = '../tmp/'+str(i)+'.npy'
    np.save(filename, responses)
    return filename

def compute_dictionary(num_workers=2):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * dictionary: numpy.ndarray of shape (K, 3F)
    '''

    start_time = time.time()
    train_data = np.load("../data/train_data.npz")
    # ----- TODO -----
    K = 255
    alpha = 250
    files = train_data['files']
    T = len(files)
    tmp_files = []
    for i in range(0,T,num_workers):
        print('Time since started:', time.time()-start_time)
        with Pool(num_workers) as pool:
            args = []
            for j in range(num_workers):
                if i+j < T:
                    image_path = '../data/'+files[i+j]
                    args.append((i+j,alpha,image_path))
            tmp_files.extend(pool.map(compute_dictionary_one_image,args))

    print('Gathering responses')
    responses = np.zeros((alpha*T,60))
    for t in range(T):
        file = tmp_files[t]
        data = np.load(file)
        responses[t*alpha:t*alpha+alpha,:] = data

    print('Starting clustering')
    kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(responses)
    dictionary = kmeans.cluster_centers_

    np.save('../dictionary.npy', dictionary)
    print('Final runtime:', time.time()-start_time)
