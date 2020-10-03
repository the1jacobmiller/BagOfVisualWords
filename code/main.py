import numpy as np
import torchvision
import util
import matplotlib.pyplot as plt
import visual_words
import visual_recog
import deep_recog
import custom
from skimage import io


if __name__ == '__main__':

    ############ Section 1 ############
    num_cores = util.get_num_CPU()
    # path_img = "../data/aquarium/sun_aztvjgubyrgvirup.jpg"
    # path_img = "../data/kitchen/sun_aaslbwtcdcwjukuo.jpg"
    path_img = "../data/windmill/sun_avgtzbktnevdkgsy.jpg"
    image = io.imread(path_img)
    image = image.astype('float')/255
    ### 1.1.2
    filter_responses = visual_words.extract_filter_responses(image)
    util.display_filter_responses(filter_responses)

    # ## 1.2.1
    points_of_interest = visual_words.get_harris_points(image, 250)
    plt.imshow(image)
    plt.scatter(points_of_interest[:, 1], points_of_interest[:, 0], s = 3, c = "y")
    plt.show()

    # ## 1.2.2
    # visual_words.compute_dictionary(num_workers=num_cores)

    # ## 1.3.1
    dictionary = np.load('dictionary.npy')
    wordmap = visual_words.get_visual_words(image, dictionary)
    util.save_wordmap(wordmap, "wordmap.png")

    # ########### Section 2 ############
    # 2.1.1
    hist = visual_recog.get_feature_from_wordmap(wordmap, dictionary.shape[0])
    plt.bar(np.arange(len(hist)), hist)
    plt.show()

    # 2.2.1
    hist = visual_recog.get_feature_from_wordmap_SPM(wordmap, 3, dictionary.shape[0])
    plt.bar(np.arange(len(hist)), hist)
    plt.show()

    # ## 2.1.1 - 2.4.1
    visual_recog.build_recognition_system(num_workers=num_cores)

    # ########### Section 3 ############
    # ## 3.1.1 - 3.1.3
    conf, accuracy = visual_recog.evaluate_recognition_system(num_workers=num_cores)
    print(conf)
    print(np.diag(conf).sum()/conf.sum())

    # 3.1.3 - SVM
    print('-----------SVM-----------')
    custom.build_recognition_system_SVM(num_workers=num_cores)
    conf, accuracy = custom.evaluate_recognition_system_SVM(num_workers=num_cores)
    print(conf)
    print(np.diag(conf).sum()/conf.sum())

    # 3.1.3 - Naive Bayes
    print('-----------Naive Bayes-----------')
    custom.build_recognition_system_Bayes(num_workers=num_cores)
    conf, accuracy = custom.evaluate_recognition_system_Bayes(num_workers=num_cores)
    print(conf)
    print(np.diag(conf).sum()/conf.sum())

    # 3.1.3 - KNN
    print('-----------KNN-----------')
    custom.build_recognition_system_KNN(num_workers=num_cores)
    conf, accuracy = custom.evaluate_recognition_system_KNN(num_workers=num_cores)
    print(conf)
    print(np.diag(conf).sum()/conf.sum())

    # 3.1.4 - IDF
    print('-----------IDF-----------')
    conf, accuracy = custom.evaluate_recognition_system_IDF(num_workers=num_cores)
    print(conf)
    print(np.diag(conf).sum()/conf.sum())

    # 3.1.4 - Bayers IDF
    print('-----------IDF-----------')
    conf, accuracy = custom.evaluate_recognition_system_Bayes_IDF(num_workers=num_cores)
    print(conf)
    print(np.diag(conf).sum()/conf.sum())

    # ########### Section 4 ############
    # # ## 4.1.1
    vgg16 = torchvision.models.vgg16(pretrained=True).double()
    vgg16.eval()
    diff = deep_recog.evaluate_deep_extractor(image, vgg16)
    print(diff)

    # ## 4.1.2
    vgg16 = torchvision.models.vgg16(pretrained=True).double()
    vgg16.eval()
    deep_recog.build_recognition_system(vgg16, 4)
    conf, accuracy = deep_recog.evaluate_recognition_system(vgg16, num_workers=4)
    print(conf)
    print(np.diag(conf).sum()/conf.sum())
