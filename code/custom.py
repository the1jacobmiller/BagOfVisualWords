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
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import pickle

from visual_recog import get_image_feature
from visual_recog import distance_to_set
from visual_recog import get_feature_from_wordmap
from visual_recog import get_feature_from_wordmap_SPM


def build_recognition_system_SVM(num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * svm model
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

    # get labels
    labels = train_data['labels']

    print('Gathering responses')
    features = np.zeros((N,M))
    for n in range(N):
        file = tmp_files[n]
        data = np.load(file)
        features[n,:] = data

    # train SVM
    svm_model = SVC(kernel = 'linear', C = 1).fit(features, labels)

    # save file
    pkl_filename = "svm_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(svm_model, file)

def build_recognition_system_Bayes(num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * naive bayes model
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

    # get labels
    labels = train_data['labels']

    print('Gathering responses')
    features = np.zeros((N,M))
    for n in range(N):
        file = tmp_files[n]
        data = np.load(file)
        features[n,:] = data

    # train naive bayes
    naive_bayes_model = GaussianNB().fit(features, labels)

    # save file
    pkl_filename = "naive_bayes_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(naive_bayes_model, file)

def build_recognition_system_KNN(num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * knn model
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

    # get labels
    labels = train_data['labels']

    print('Gathering responses')
    features = np.zeros((N,M))
    for n in range(N):
        file = tmp_files[n]
        data = np.load(file)
        features[n,:] = data

    # train knn bayes
    knn_model = KNeighborsClassifier(n_neighbors = 3).fit(features, labels)

    # save file
    pkl_filename = "knn_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(knn_model, file)

def evaluate_recognition_system_IDF(num_workers=2):
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
    IDF = np.load('idf.npy')
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
                    args.append((image_path,i+j,dictionary,layer_num,K,IDF))
            feature_tmp_files = pool.map(get_image_feature_IDF,args)
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

def evaluate_recognition_system_SVM(num_workers=2):
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
    dictionary = np.load("dictionary.npy")

    # Load from file
    pkl_filename = "svm_model.pkl"
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)

    layer_num = 3
    K = dictionary.shape[0]
    C = np.zeros((8,8))
    M = int(K*(4**layer_num-1)/3)

    # extract histograms of features
    files = test_data['files']
    N = len(files)
    feature_tmp_files = []
    for i in range(0,N,num_workers):
        print(100*i/N, 'percent done')
        print('Time since started:', time.time()-start_time)
        with Pool(num_workers) as pool:
            args = []
            for j in range(num_workers):
                if i+j < N:
                    image_path = '../data/'+files[i+j]
                    args.append((image_path,i+j,dictionary,layer_num,K))
            feature_tmp_files.extend(pool.map(get_image_feature,args))

    features = np.zeros((N,M))
    for i in range(len(feature_tmp_files)):
        data = np.load(feature_tmp_files[i])
        features[i,:] = data

    test_labels = test_data['labels']
    predictions = pickle_model.predict(features)
    for i in range(len(test_labels)):
        label = test_labels[i]
        pred = predictions[i]
        C[label,pred] += 1

    os.system('rm -r tmp_features')

    return C,np.diag(C).sum()/C.sum()

def evaluate_recognition_system_Bayes(num_workers=2):
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
    dictionary = np.load("dictionary.npy")

    # Load from file
    pkl_filename = "naive_bayes_model.pkl"
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)

    layer_num = 3
    K = dictionary.shape[0]
    C = np.zeros((8,8))
    M = int(K*(4**layer_num-1)/3)

    # extract histograms of features
    files = test_data['files']
    N = len(files)
    feature_tmp_files = []
    for i in range(0,N,num_workers):
        print(100*i/N, 'percent done')
        print('Time since started:', time.time()-start_time)
        with Pool(num_workers) as pool:
            args = []
            for j in range(num_workers):
                if i+j < N:
                    image_path = '../data/'+files[i+j]
                    args.append((image_path,i+j,dictionary,layer_num,K))
            feature_tmp_files.extend(pool.map(get_image_feature,args))

    features = np.zeros((N,M))
    for i in range(len(feature_tmp_files)):
        data = np.load(feature_tmp_files[i])
        features[i,:] = data

    test_labels = test_data['labels']
    predictions = pickle_model.predict(features)
    for i in range(len(test_labels)):
        label = test_labels[i]
        pred = predictions[i]
        C[label,pred] += 1

    os.system('rm -r tmp_features')

    return C,np.diag(C).sum()/C.sum()

def evaluate_recognition_system_KNN(num_workers=2):
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
    dictionary = np.load("dictionary.npy")

    # Load from file
    pkl_filename = "knn_model.pkl"
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)

    layer_num = 3
    K = dictionary.shape[0]
    C = np.zeros((8,8))
    M = int(K*(4**layer_num-1)/3)

    # extract histograms of features
    files = test_data['files']
    N = len(files)
    feature_tmp_files = []
    for i in range(0,N,num_workers):
        print(100*i/N, 'percent done')
        print('Time since started:', time.time()-start_time)
        with Pool(num_workers) as pool:
            args = []
            for j in range(num_workers):
                if i+j < N:
                    image_path = '../data/'+files[i+j]
                    args.append((image_path,i+j,dictionary,layer_num,K))
            feature_tmp_files.extend(pool.map(get_image_feature,args))

    features = np.zeros((N,M))
    for i in range(len(feature_tmp_files)):
        data = np.load(feature_tmp_files[i])
        features[i,:] = data

    test_labels = test_data['labels']
    predictions = pickle_model.predict(features)
    for i in range(len(test_labels)):
        label = test_labels[i]
        pred = predictions[i]
        C[label,pred] += 1

    os.system('rm -r tmp_features')

    return C,np.diag(C).sum()/C.sum()

def evaluate_recognition_system_Bayes_IDF(num_workers=2):
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
    dictionary = np.load("dictionary.npy")
    IDF = np.load("idf.npy")

    # Load from file
    pkl_filename = "naive_bayes_model.pkl"
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)

    layer_num = 3
    K = dictionary.shape[0]
    C = np.zeros((8,8))
    M = int(K*(4**layer_num-1)/3)

    # extract histograms of features
    files = test_data['files']
    N = len(files)
    feature_tmp_files = []
    for i in range(0,N,num_workers):
        print(100*i/N, 'percent done')
        print('Time since started:', time.time()-start_time)
        with Pool(num_workers) as pool:
            args = []
            for j in range(num_workers):
                if i+j < N:
                    image_path = '../data/'+files[i+j]
                    args.append((image_path,i+j,dictionary,layer_num,K,IDF))
            feature_tmp_files.extend(pool.map(get_image_feature_IDF,args))

    features = np.zeros((N,M))
    for i in range(len(feature_tmp_files)):
        data = np.load(feature_tmp_files[i])
        features[i,:] = data

    test_labels = test_data['labels']
    predictions = pickle_model.predict(features)
    for i in range(len(test_labels)):
        label = test_labels[i]
        pred = predictions[i]
        C[label,pred] += 1

    os.system('rm -r tmp_features')

    return C,np.diag(C).sum()/C.sum()

def get_image_feature_IDF(args):
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
    file_path, i, dictionary, layer_num, K, IDF = args
    image = skimage.io.imread(file_path)
    image = image.astype('float')/255
    wordmap = visual_words.get_visual_words(image, dictionary)
    features = get_feature_from_wordmap_SPM_IDF(wordmap, layer_num, K, IDF)

    # save filter features to a temp file
    if not os.path.isdir('tmp_features'):
        os.system('mkdir tmp_features')
    filename = 'tmp_features/'+str(i)+'.npy'
    np.save(filename, features)
    return filename

def compute_IDF(features,K):
    T = features.shape[0]
    layer1_features = features[:,-K:]
    IDF = np.zeros((1,K))

    for k in range(K):
        counts = layer1_features[:,k]
        IDF[0,k] = np.log(T/np.argwhere(counts>0).shape[0])

    np.save('idf.npy', IDF)

def get_feature_from_wordmap_IDF(wordmap, dict_size, IDF):
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
    hist = hist*IDF[0]

    return hist / np.linalg.norm(hist, ord=1)

def get_feature_from_wordmap_SPM_IDF(wordmap, layer_num, dict_size, IDF):
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
            hist_layer = get_feature_from_wordmap_IDF(cell,dict_size,IDF)
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
                    hist = get_feature_from_wordmap_IDF(cell,dict_size,IDF)
                    hist_layer[cell_num*dict_size:cell_num*dict_size+dict_size] = hist
                    cell_num += 1
            norm_factor = 1./(cells*cells)
            weight = 2.**(l-layer_num)
            hist_layer = norm_factor*weight*hist_layer
            hist_all[hist_all_index:hist_all_index+hist_layer.shape[0]] = hist_layer
            hist_all_index += hist_layer.shape[0]
    return hist_all
