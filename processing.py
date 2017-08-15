#coding=utf-8
import os
# import cv2
from PIL import Image
import numpy as np
from keras.models import Model
from keras.layers import Input,GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator



f_train = open("./data_train.txt",'rb')
# for i in f_train:
#     print i.split()
import pandas as pd
df = pd.read_table("./data_train_image.txt",header=None,sep=' ')
df2 = pd.DataFrame({'image_id':df[0],'class_id':df[1]})
# print df2
df3 = pd.read_table("./val.txt",header=None,sep=' ')
df4 = pd.DataFrame({'image_id':df3[0],'class_id':df3[1]})
train_df = pd.DataFrame(pd.concat([df2,df4],ignore_index=True),columns=['image_id','class_id'])
print train_df
#删除异常数据
del_list_1 = []
del_list_2 = []
list_1 = [list for list in os.listdir("./train")]
list_2 = [list for list in os.listdir("./test1")]
for i in range(len(list_1)):
    if len(train_df[train_df['image_id']==list_1[i][0:-4]]['class_id'].values)!=1:
        del_list_1.append(list_1[i])
for i in range(len(list_2)):
    if len(train_df[train_df['image_id']==list_2[i][0:-4]]['class_id'].values)!=1:
        del_list_2.append(list_2[i])
for d in range(len(del_list_1)):
    if os.path.exists('./train/'+del_list_1[d]):
        os.remove('./train/'+del_list_1[d])
for d in range(len(del_list_2)):
    if os.path.exists('./test1/'+del_list_2[d]):
        os.remove('./test1/'+del_list_2[d])

list_1 = [list for list in os.listdir("./train")]
# print list_1[0][0:-4]
# print train_df[train_df['image_id']==list_1[0][0:-4]]['class_id'].values
# exit()
list_2 = [list for list in os.listdir("./test1")]
# print
n = len(list_1)+len(list_2)
X = np.zeros((n,3,224,224),dtype=np.uint8)
y = np.zeros((n,),dtype=np.uint8)
from keras.preprocessing.image import img_to_array

for i in range(len(list_1)):
    img = Image.open('./train/'+list_1[i])
    X[i] = img_to_array(img)#np.asarray(img.resize((224, 224))).reshape((3,224,224))
    # print train_df[train_df['image_id']==list_1[i][0:-4]]['class_id'].values
    print list_1[i]
    y[i] = train_df[train_df['image_id']==list_1[i][0:-4]]['class_id'].values
for j in range(len(list_2)):
    img = Image.open('./test1/' + list_2[j])
    X[j+len(list_1)] = img_to_array(img)#np.asarray(img.resize((224, 224))).reshape((3,224,224))
    print list_2[j]
    y[j+len(list_1)] = train_df[train_df['image_id'] == list_2[j][0:-4]]['class_id'].values
# import pickle
# with open("train.pkl",'wb') as f:
#     pickle.dump(X,f)
#     pickle.dump(y,f,-1)
#     f.close()
# f = open('train.pkl','rb')
# data_X = pickle.load(f)
# print data_X.shape
# data_Y = pickle.load(f)
# print data_Y.shape

# from keras.applications.resnet50 import ResNet50
# from keras.applications.xception import Xception
# from keras.applications.inception_v3 import InceptionV3
# model_1 = ResNet50(include_top=False,weights='imagenet')
# model = Xception(include_top=False,weights='imagenet')
# model_2 = InceptionV3(include_top=False,weights='imagenet')
#from keras.preprocessing.image import ImageDataGenerator
#from keras.utils import np_utils
#y = np_utils.to_categorical(y,y.max())

#gen = ImageDataGenerator(
 #   featurewise_center=True

#)
#train_gen = gen.flow()
