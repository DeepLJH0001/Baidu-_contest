#coding=utf-8
import h5py
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# print os.listdir("./feature_")
# exit()
X_train = []
X_test = []
for filename in os.listdir("./feature_"):
    with h5py.File('./feature_/'+filename,'r') as h:
        X_train.append(np.array(h['train']))
        X_test.append(np.array(h['test']))
Train_X = np.hstack((X_train[0],X_train[1],X_train[2],X_train[3],X_train[4]))#5个都进行提取
Test_X = np.hstack((X_test[0],X_test[1],X_test[2],X_test[3],X_test[4]))
X_train = []
X_test = []
y = np.array([0,])
for filename in  ["inceptionV3_feature.h5","MobileNet_feature.h5","res_feature.h5","VGG19_feature.h5","Xception_feature.h5",]:
    with h5py.File(filename,'r') as h:
        X_train.append(np.array(h['train']))
        # X_test.append(np.array(h['test']))
        y =  np.array(h['label'])
# Train_X = np.vstack((Train_X,np.hstack((X_train[0],X_train[1],X_train[2],X_train[3],X_train[4]))))
# Test_X = np.hstack((X_test[0],X_test[1],X_test[2],X_test[3],X_test[4]))
# Test_X = np.vstack((Test_X,np.hstack((X_test[0],X_test[1],X_test[2],X_test[3],X_test[4]))))
    # exit()

scale = MinMaxScaler(feature_range=(0,1))
scale.fit_transform(Train_X)
scale.transform(Test_X)
df = pd.read_csv("./df.csv")
label = df['index_id'].values
class_id = df['class_id'].values

label = label.reshape((label.shape[0],1))
y = y.reshape((y.shape[0],1))
dict1 = dict()
dict2 = dict()
class_id = np.unique(class_id)
index_id = np.arange(0,100,1)
for i in range(index_id.shape[0]):
    dict1[index_id[i]] = class_id[i]
for i in range(index_id.shape[0]):
    dict2[class_id[i]] = index_id[i]
for i in range(y.shape[0]):
    y[i][0] = dict2[y[i][0]]
# label = np.vstack((label,y))
label = label.reshape((label.shape[0],))
print (Train_X.shape,Test_X.shape,label.shape)
class_weight = np.array([ 1.65876777,  1.4       ,  1.45228216,  1.47058824,  1.4  ,
        1.89189189,  1.60550459,  4.72972973,  1.75879397,  4.86111111,
        4.16666667,  7.        ,  1.40562249,  6.48148148,  3.5       ,
        6.03448276,  1.75      ,  1.75      ,  1.40562249,  4.48717949,
        1.41129032,  1.96629213,  1.77664975,  1.4       ,  1.69082126,
        1.4       ,  5.2238806 ,  1.4       ,  1.41129032,  1.4       ,
        2.09580838,  3.27102804,  1.42276423,  1.4       ,  3.72340426,
        1.40562249,  1.4       ,  1.40562249,  1.4       ,  3.39805825,
        5.14705882,  1.44032922,  3.36538462,  5.73770492,  2.77777778,
        2.09580838,  1.40562249,  1.41129032,  1.4       ,  1.40562249,
        1.40562249,  2.51798561,  1.4       ,  1.97740113,  1.41700405,
        1.54867257,  1.79487179,  1.66666667,  2.71317829,  1.89189189,
        1.40562249,  1.41129032,  1.4       ,  1.41129032,  1.40562249,
        3.24074074,  1.60550459,  2.13414634,  2.65151515,  3.80434783,
        2.1875    ,  2.04678363,  1.40562249,  2.86885246,  3.72340426,
        1.66666667,  1.40562249,  2.71317829,  4.32098765,  3.64583333,
        3.18181818,  5.73770492,  1.63551402,  1.2195122 ,  6.60377358,
        1.52173913,  5.07246377,  1.40562249,  1.88172043,  1.41700405,
        1.40562249,  1.40562249,  1.4       ,  1.4       ,  1.54867257,
        1.        ,  2.69230769,  1.2962963 ,  1.4       ,  6.25      ], dtype = np.float32)

from keras.layers import Dropout,Dense,Input,BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential,Model
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.utils import np_utils
from keras.optimizers import Adam,RMSprop
from keras.regularizers import l2

label = np_utils.to_categorical(label,100)
input_tensor = Input(Train_X.shape[1:])
from keras.constraints import maxnorm
model = Sequential()
model.add(Dropout(0.85,input_shape=(Train_X.shape[1:])))
model.add(BatchNormalization())
# model.add(PReLU())
model.add(Dense(1024))
model.add(PReLU())
model.add(Dropout(0.8))
# model.add(Dense(512))
# model.add(PReLU())
# model.add(Dropout(0.5))
model.add(Dense(100,activation='softmax'))

# adam = Adam(lr=0.0001)
rmsprop = RMSprop(lr=0.0001)
from keras.callbacks import ReduceLROnPlateau
model.compile(optimizer=rmsprop,loss="categorical_crossentropy", metrics=['accuracy'])
model.fit(Train_X,label,nb_epoch=600,batch_size=1024,verbose=2,shuffle=True,#class_weight=class_weight,#validation_split=0.2,
          callbacks=[ ModelCheckpoint("./best.h",monitor='acc',save_best_only=True,verbose=2),
                   # EarlyStopping(monitor='val_loss',patience=50,verbose=2)
                    ReduceLROnPlateau(patience=50,factor=0.2,cooldown=50)
            ]
          )
model.load_weights("./best.h")
print (model.evaluate(Train_X,label,verbose=2))
result = model.predict_classes(Test_X,verbose=2)
for i in range(result.shape[0]):
    result[i] = dict1[result[i]]
list_2 = [list for list in os.listdir("./ssdimage")]

with open("./result.txt",'wb') as f:
    for i in range(len(list_2)):
        f.write('%d\t%s\n'%(result[i],list_2[i][0:-4]))
    f.close()
