import numpy as np
import skimage
import multiprocessing
import threading
import queue
import os,time
import math
import visual_words
import matplotlib.pyplot as plt
from multiprocessing import Pool
import pickle


def build_recognition_system(num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N, M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K, 3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    print('Building recognition system')
    start_time = time.time()
    train_data = np.load("../data/train_data.npz")
    dictionary = np.load("dictionary.npy")

    # define sizes
    layer_num = 3
    K = dictionary.shape[0]
    M = int(K*(4**layer_num-1)/3)

    # extract histograms of features
    files = train_data['files']
    N = len(files)
    tmp_files = []
    for i in range(0,N,num_workers):
        print(100*i/N, 'percent done')
        print('Time since started:', time.time()-start_time)
        with Pool(num_workers) as pool:
            args = []
            for j in range(num_workers):
                if i+j < N:
                    image_path = '../data/'+files[i+j]
                    args.append((image_path,i+j,dictionary,layer_num,K))
            tmp_files.extend(pool.map(get_image_feature,args))

    print('Gathering responses')
    features = np.zeros((N,M))
    for n in range(N):
        file = tmp_files[n]
        data = np.load(file)
        features[n,:] = data

    # get labels
    labels = train_data['labels']

    # save numpy file
    np.savez('trained_system.npz', dictionary, features, labels, layer_num)

def evaluate_recognition_system(num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8, 8)
    * accuracy: accuracy of the evaluated system
    '''

    print('Evaluating recognition system')
    start_time = time.time()
    test_data = np.load("../data/test_data.npz")
    trained_system = np.load("trained_system.npz")
    dictionary = trained_system['arr_0']
    features = trained_system['arr_1']
    labels = trained_system['arr_2']
    layer_num = trained_system['arr_3']

    C = np.zeros((8,8))
    K = dictionary.shape[0]

    # extract histograms of features
    files = test_data['files']
    N = len(files)
    feature_tmp_files = []
    dist_tmp_files = []
    for i in range(0,N,num_workers):
        print(100*i/N, 'percent done')
        print('Time since started:', time.time()-start_time)
        with Pool(num_workers) as pool:
            args = []
            for j in range(num_workers):
                if i+j < N:
                    image_path = '../data/'+files[i+j]
                    args.append((image_path,i+j,dictionary,layer_num,K))
            feature_tmp_files = pool.map(get_image_feature,args)
            args = []
            for j in range(num_workers):
                if i+j < N:
                    wordmap = np.load(feature_tmp_files[j])
                    args.append((wordmap,features,i+j))
            dist_tmp_files.extend(pool.map(distance_to_set,args))

    test_labels = test_data['labels']
    for i in range(len(dist_tmp_files)):
        label = test_labels[i]
        dists = np.load(dist_tmp_files[i])
        pred = labels[np.argmax(dists)]
        C[label,pred] += 1

    os.system('rm -r tmp_dist')
    os.system('rm -r tmp_features')

    return C,np.diag(C).sum()/C.sum()

def get_image_feature(args):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''
    file_path, i, dictionary, layer_num, K = args
    image = skimage.io.imread(file_path)
    image = image.astype('float')/255
    wordmap = visual_words.get_visual_words(image, dictionary)
    features = get_feature_from_wordmap_SPM(wordmap, layer_num, K)

    # save filter features to a temp file
    if not os.path.isdir('tmp_features'):
        os.system('mkdir tmp_features')
    filename = 'tmp_features/'+str(i)+'.npy'
    np.save(filename, features)
    return filename

def distance_to_set(args):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N, K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''

    word_hist, histograms, i = args
    dists = np.sum(np.minimum(word_hist,histograms),axis=1)

    # save filter features to a temp file
    if not os.path.isdir('tmp_dist'):
        os.system('mkdir tmp_dist')
    filename = 'tmp_dist/'+str(i)+'.npy'
    np.save(filename, dists)
    return filename

def get_feature_from_wordmap(wordmap, dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H, W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    hist = np.zeros((dict_size,))
    unique,counts = np.unique(wordmap.flatten(), return_counts=True)
    hist[unique] = counts

    return hist / np.linalg.norm(hist, ord=1)

def get_feature_from_wordmap_SPM(wordmap, layer_num, dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H, W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''

    hist_size = int(dict_size*(4**layer_num-1)/3)
    hist_all = np.zeros((hist_size,))
    hist_all_index = 0
    rows,cols = wordmap.shape

    for l in range(layer_num-1,-1,-1):
        if l == 0:
            hist_layer = get_feature_from_wordmap(cell,dict_size)
            weight = 2.**(-layer_num+1)
            hist_layer = weight*hist_layer
            hist_all[hist_all_index:hist_all_index+hist_layer.shape[0]] = hist_layer
            hist_all_index += hist_layer.shape[0]
        else:
            cells = 2**l
            cell_rows = int(rows/cells)
            cell_cols = int(cols/cells)
            hist_layer = np.zeros((cells*cells*dict_size,))
            cell_num = 0
            for m in range(0,rows-cell_rows+1,cell_rows):
                for n in range(0,cols-cell_cols+1,cell_cols):
                    cell = wordmap[m:m+cell_rows,n:n+cell_cols]
                    hist = get_feature_from_wordmap(cell,dict_size)
                    hist_layer[cell_num*dict_size:cell_num*dict_size+dict_size] = hist
                    cell_num += 1
            norm_factor = 1./(cells*cells)
            weight = 2.**(l-layer_num)
            hist_layer = norm_factor*weight*hist_layer
            hist_all[hist_all_index:hist_all_index+hist_layer.shape[0]] = hist_layer
            hist_all_index += hist_layer.shape[0]
    # plt.bar(np.arange(len(hist_all)), hist_all)
    # plt.show()
    return hist_all
