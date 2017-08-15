from keras.models import Model
from keras.layers import Input,GlobalAveragePooling2D
from keras.preprocessing.image import load_img,img_to_array
import pandas as pd
import numpy as np
# from keras.preprocessing.image import ImageDataGenerator
import os
import h5py
def get_model(type = 0):
    if type==0:
        input_x = Input((3,224,224))#theano
        from keras.applications.resnet50 import ResNet50,preprocess_input
        model = ResNet50(input_tensor=input_x,weights='imagenet',include_top=False)
        func = preprocess_input

    elif type == 1:
        input_x = Input((3,299,299))#theano
        from keras.applications.inception_v3 import InceptionV3,preprocess_input
        model = InceptionV3(input_tensor=input_x,weights='imagenet',include_top=False)
        func = preprocess_input

    elif type == 2:
        input_x = Input((299,299,3))#tensorflow
        from keras.applications.xception import Xception,preprocess_input
        model = Xception(input_tensor=input_x,weights='imagenet',include_top=False)
        func = preprocess_input

    elif type == 3:
        input_x = Input((299,299,3))#tensorflow
        from keras.applications.mobilenet import MobileNet,preprocess_input
        model = MobileNet(input_tensor=input_x,weights='imagenet',include_top=False)
        func = preprocess_input

    elif type == 4:
        input_x = Input((3,299,299))#theano
        from keras.applications.vgg19 import VGG19,preprocess_input
        model = VGG19(input_tensor=input_x,weights='imagenet',include_top=False)
        func = preprocess_input

    return model,preprocess_input
if __name__ == '__main__':
    base_model,preprocess_input = get_model(type=2)
    target_size = (299,299)
    # target_size = (224,224)#resnet only
    # save_name = 'res50_feature_ssd.h5'
    # save_name = 'inceptionv3_feature_ssd.h5'
    save_name = "Xception_feature_ssd.h5"
    # save_name = 'mobilenet_feature_ssd.h5'
    # save_name = 'vgg19_feature_ssd.h5'
    save_path = './feature_/'
    model = Model(input=base_model.input,output=GlobalAveragePooling2D()(base_model.output))

    BATCHSIZE = 1024

    df = pd.read_csv("./df.csv")
    train_feature = np.array([0, ])
    test_feature = np.array([0, ])
    for i in range(int(df.shape[0]/BATCHSIZE)+1):
        imgs = []
        for j in range(BATCHSIZE):
            # print i*BATCHSIZE+j
            # print df.iloc[i*BATCHSIZE+j,0]
            imgs.append(img_to_array(load_img(df.iloc[i*BATCHSIZE+j,0],target_size=target_size)))
            if i*BATCHSIZE+j == df.shape[0]-1:
                break
        imgs = np.asarray(imgs)
        if i == 0:
            train_feature = model.predict(preprocess_input(imgs))
        else:
            train_feature = np.vstack([train_feature, model.predict(preprocess_input(imgs))])
    print (train_feature.shape)
    list_2 = ['./ssdimage/'+list for list in os.listdir("./ssdimage")]
    for i in range(int(len(list_2)/BATCHSIZE)+1):
        imgs = []
        for j in range(BATCHSIZE):
            imgs.append(img_to_array(load_img(list_2[i * BATCHSIZE + j], target_size=target_size)))
            if i * BATCHSIZE + j == len(list_2)-1:
                break
        imgs = np.asarray(imgs)
        if i == 0:
            test_feature = model.predict(preprocess_input(imgs))
        else:
            test_feature = np.vstack([test_feature, model.predict(preprocess_input(imgs))])
    print (test_feature.shape,len(list_2))
    with h5py.File(save_name) as h:
        h.create_dataset("train", data=train_feature)
        h.create_dataset("test", data=test_feature)

