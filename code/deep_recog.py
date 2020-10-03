import numpy as np
import multiprocessing
import threading
import queue
import os,time
import torch
import skimage.transform
import torchvision.transforms
import util
import network_layers
from multiprocessing import Pool
from skimage import io

torch.set_num_threads(1)  # without this line, pytroch forward pass will hang with multiprocessing

def evaluate_deep_extractor(img, vgg16):
    '''
    Evaluates the deep feature extractor for a single image.

    [input]
    * image: numpy.ndarray of shape (H,W,3)
    * vgg16: prebuilt VGG-16 network.

    [output]
    * diff: difference between the two feature extractor's result
    '''
    vgg16_weights = util.get_VGG16_weights()
    img_torch = preprocess_image(img)

    feat = network_layers.extract_deep_feature(np.transpose(img_torch.numpy(), (1,2,0)), vgg16_weights)

    with torch.no_grad():
        vgg_classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-3])
        vgg_feat_feat = vgg16.features(img_torch[None, ])
        vgg_feat_feat = vgg_classifier(vgg_feat_feat.flatten())

    return np.sum(np.abs(vgg_feat_feat.numpy() - feat))

def build_recognition_system(vgg16, num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * vgg16: prebuilt VGG-16 network.
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N, K)
    * labels: numpy.ndarray of shape (N)
    '''

    train_data = np.load("../data/train_data.npz")

    print('Building deep recognition system')
    start_time = time.time()

    # extract features
    files = train_data['files']
    N = len(files)
    K = 4096
    tmp_files = []
    for i in range(0,N,num_workers):
        print(100*i/N, 'percent done')
        print('Time since started:', time.time()-start_time)
        with Pool(num_workers) as pool:
            args = []
            for j in range(num_workers):
                if i+j < N:
                    image_path = '../data/'+files[i+j]
                    args.append((i+j,image_path,vgg16))
            tmp_files.extend(pool.map(get_image_feature,args))

    print('Gathering responses')
    features = np.zeros((N,K))
    for n in range(N):
        file = tmp_files[n]
        data = np.load(file)
        features[n,:] = data

    # get labels
    labels = train_data['labels']

    # save numpy file
    np.savez('trained_system_deep.npz', features, labels)

def evaluate_recognition_system(vgg16, num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * vgg16: prebuilt VGG-16 network.
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8, 8)
    * accuracy: accuracy of the evaluated system
    '''

    test_data = np.load("../data/test_data.npz")
    trained_system = np.load("trained_system_deep.npz")

    print('Evaluating deep recognition system')
    start_time = time.time()

    features = trained_system['arr_0']
    labels = trained_system['arr_1']

    C = np.zeros((8,8))

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
                    args.append((i+j,image_path,vgg16))
            feature_tmp_files = pool.map(get_image_feature,args)
            args = []
            for j in range(num_workers):
                if i+j < N:
                    img_features = np.load(feature_tmp_files[j])
                    args.append((img_features,features,i+j))
            dist_tmp_files.extend(pool.map(distance_to_set,args))

    test_labels = test_data['labels']
    for i in range(len(dist_tmp_files)):
        label = test_labels[i]
        dists = np.load(dist_tmp_files[i])
        pred = labels[np.argmax(dists)]
        C[label,pred] += 1

    os.system('rm -r tmp_deep_dist')
    os.system('rm -r tmp_deep_features')

    return C,np.diag(C).sum()/C.sum()

def preprocess_image(image):
    '''
    Preprocesses the image to load into the prebuilt network.

    [input]
    * image: numpy.ndarray of shape (H, W, 3)

    [output]
    * image_processed: torch.Tensor of shape (3, H, W)
    '''

    # # resize image to 224x224 and normalize
    # mean = np.array([0.485,0.456,0.406])
    # std = np.array([0.229,0.224,0.225])
    # image = skimage.transform.resize(image, (224,224))
    # image = (image-mean)/std

    image = skimage.transform.resize(image, (224,224))
    tensor = torch.from_numpy(image.transpose((2, 0, 1)))
    return tensor

def get_image_feature(args):
    '''
    Extracts deep features from the prebuilt VGG-16 network.
    This is a function run by a subprocess.
    [input]
    * i: index of training image
    * image_path: path of image file
    * vgg16: prebuilt VGG-16 network.

    [output]
    * feat: evaluated deep feature
    '''

    i, image_path, vgg16 = args

    # load image and weights
    image = io.imread(image_path)
    image = image.astype('float')/255
    img_torch = preprocess_image(image)

    with torch.no_grad():
        vgg_classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-3])
        vgg_feat_feat = vgg16.features(img_torch[None, ])
        vgg_feat_feat = vgg_classifier(vgg_feat_feat.flatten())

    # save features to a temp file
    if not os.path.isdir('tmp_deep_features'):
        os.system('mkdir tmp_deep_features')
    filename = 'tmp_deep_features/'+str(i)+'.npy'
    np.save(filename, vgg_feat_feat)
    return filename

def distance_to_set(args):
    '''
    Compute distance between a deep feature with all training image deep features.

    [input]
    * feature: numpy.ndarray of shape (K)
    * train_features: numpy.ndarray of shape (N, K)

    [output]
    * dist: numpy.ndarray of shape (N)
    '''

    feature, train_features, i = args
    dists = -1.0*np.sqrt(np.sum(np.square(feature-train_features),axis=1))

    # save filter features to a temp file
    if not os.path.isdir('tmp_deep_dist'):
        os.system('mkdir tmp_deep_dist')
    filename = 'tmp_deep_dist/'+str(i)+'.npy'
    np.save(filename, dists)
    return filename
