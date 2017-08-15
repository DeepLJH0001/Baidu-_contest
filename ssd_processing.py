#coding=utf-8
import pandas as pd
import os
import numpy as np
import shutil
#ssd 没能box出来的狗的image复制到ssd文件夹
lack_test_image = pd.read_table('./lackimage.txt',header=None)
print lack_test_image.shape
for i in range(lack_test_image.shape[0]):
    shutil.copy('./image/'+lack_test_image[0].values[i],"./ssdimage/")
lack_train_image = pd.read_table('./lack.txt', header=None)
for i in range(lack_train_image.shape[0]):
    shutil.copy('./train/'+lack_train_image[0].values[i],"./ssdtrain/")
    # print lack_test_image[0].values[i]
# exit()
# exit()
list_1 = ['./ssdtrain/'+list for list in os.listdir("./ssdtrain")]#.extend(
list_3 = ['./ssdtest1/'+list for list in os.listdir('./ssdtest1')]#[0:10]
for i in list_3:
    list_1.append(i)
print len(list_1)
# list_2 = [list for list in os.listdir("./image")]
df = pd.read_table("./data_train_image.txt",header=None,sep=' ')
df2 = pd.DataFrame({'image_id':df[0],'class_id':df[1]})
df3 = pd.read_table("./val.txt",header=None,sep=' ')
df4 = pd.DataFrame({'image_id':df3[0],'class_id':df3[1]})
train_df = pd.DataFrame(pd.concat([df2,df4],ignore_index=True),columns=['image_id','class_id'])
unique = np.unique(train_df['class_id'].values)
_dict = dict()
values = np.arange(0,100,1)
for i in range(values.shape[0]):
    _dict[unique[i]] = values[i]
index_id = []
path = []
image_id = []
class_id = []
for j in range(len(list_1)):
    path.append(list_1[j])
    image_id.append(list_1[j][11:-4])
    s = train_df[train_df['image_id']==list_1[j][11:-4]]['class_id'].values[0]
    class_id.append(s)
    index_id.append(_dict[s])
path = np.array(path)
image_id = np.array(image_id)
class_id = np.array(class_id)
index_id = np.array(index_id)
train_df = pd.DataFrame({"path":path,"image_id":image_id,"class_id":class_id,"index_id":index_id})
train_df = pd.DataFrame(train_df,columns=["path","image_id","class_id","index_id"])
train_df.to_csv("./df.csv",index=False)