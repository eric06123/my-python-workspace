import glob
import os
import cv2
import numpy as np
from dataGenerator import DataGeneratorHomographyNet
import os
import glob
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import tensorflow as tf
from homographyNet import HomographyNet
import dataGenerator as dg

img_dir = "cut2plate_img/"
img_ext = ".jpg"
model_dir = "model/20210122-214300/model.h5"

model = keras.models.load_model(model_dir)

model.summary()
img_paths = glob.glob(os.path.join(img_dir, '*' + img_ext))
print(img_paths)


img = cv2.imread(img_paths[0])
cv2.imshow('i',img)
cv2.waitKey(0)

print(img.shape)
dg = DataGeneratorHomographyNet(img_paths, input_dim=(360, 360))

data, label = dg.__getitem__(0)
#print(data)
print(data.shape)

print("true",label)
print(label.shape)
m = model.predict(data)
print("predict",m)
print(m.shape)

k()

'''

#k()

#for idx in range(dg.batch_size):
for idx in range(0,1):
    cv2.imshow("orig", data[idx, :, :, 0])
    cv2.imshow("transformed", data[idx, :, :, 1])
    cv2.waitKey(0)
'''
## start training
batch_size = 2
verbose=1

input_size = (360, 360, 2)
#划分训练集和验证集，验证集搞小一点，不然每个epoch跑完太慢了
train_idx, val_idx = train_test_split(img_paths, test_size=0.01)
#拿到训练数据
train_dg = dg.DataGeneratorHomographyNet(train_idx, input_dim=input_size[0:2], batch_size=batch_size)
#拿到既定事实的标签
val_dg = dg.DataGeneratorHomographyNet(val_idx, input_dim=input_size[0:2], batch_size=batch_size)
#对于神经网络来说这个鬼一样的图就是输入，它自己从这幅图的左边和右边学习出单应性矩阵，神奇吧？
#修正网络输入头
homo_net = HomographyNet(input_size)
#实例化网络结构
model = homo_net.build_model()
#输出模型
model.summary()


## check point
checkpoint = ModelCheckpoint(
    os.path.join(model_dir, 'model.h5'),
    monitor='val_loss',
    verbose=verbose,
    save_best_only=True,
    save_weights_only=False,
    mode='auto'
)

## train model
history = model.fit_generator(train_dg, 
                              validation_data = val_dg,
                              #steps_per_epoch = 32, 
                              callbacks = [checkpoint], 
                              epochs = 10, 
                              verbose = 1)


model.save(os.path.join(model_dir, 'model.h5'))



## 參考 https://blog.csdn.net/Andrwin/article/details/105517806
## 參考 https://github.com/4nthon/HomographyNet_Keras