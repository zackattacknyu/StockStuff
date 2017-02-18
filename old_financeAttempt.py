import numpy as np
#import dicom
#import glob
from matplotlib import pyplot as plt
import os
import csv
#import cv2
import time
import datetime
# import mxnet as mx
# import pandas as pd
# from sklearn import cross_validation
# import xgboost as xgb
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Convolution3D, MaxPooling3D
from keras.utils import np_utils
from keras import backend as K

from keras.applications.resnet50 import ResNet50
import scipy.io as sio
from scipy.misc import imresize
from sklearn import cross_validation
import xgboost as xgb
from keras.utils import np_utils
import glob
from sklearn.metrics import accuracy_score
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Convolution3D, MaxPooling3D
from keras.utils import np_utils
from keras import backend as K

import scipy.io as sio
from scipy.misc import imread, imresize, imsave
import time
import datetime

from sklearn.decomposition import PCA,KernelPCA
from convnetskeras.convnets import preprocess_image_batch, convnet
from sklearn.cross_validation import train_test_split
from sklearn import random_projection
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from scipy.misc import imread, imresize, imsave
from sklearn import datasets, linear_model
from convnetskeras.customlayers import convolution2Dgroup, crosschannelnormalization, \
    splittensor, Softmax4D
from convnetskeras.imagenet_tool import synset_to_id, id_to_synset,synset_to_dfs_ids
import csv

"""
origNet = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
net = Model(input=origNet.input,output=origNet.get_layer('flatten_1').output)
"""
patFile = "/home/zdestefa/financeProj/TrainTestData2.mat"
curMATcontent = sio.loadmat(patFile)
x = curMATcontent["XtrainOther"]
y2Given = curMATcontent["yDataCategory"]
yGiven = curMATcontent["Ydata"]
y = np.reshape(yGiven,(498))
y2 = np.reshape(y2Given,(498))+1

#trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x, y, random_state=42, stratify=y,test_size=0.20)
trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x, y, random_state=42, test_size=0.20)
trn_x2, val_x2, trn_y2, val_y2 = cross_validation.train_test_split(x, y2, random_state=42, test_size=0.20)
clf = xgb.XGBRegressor(max_depth=10, n_estimators=300, min_child_weight=9,learning_rate=0.05,nthread=8,subsample=0.80,colsample_bytree=0.80,seed=4242)
#clf = xgb.XGBClassifier(max_depth=10, n_estimators=300, min_child_weight=9,learning_rate=0.05,nthread=8,subsample=0.80,colsample_bytree=0.80,seed=4242)

#TODO: INVESTIGATE FURTHUR THE RESULTS FROM HERE
clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='rmse', early_stopping_rounds=100)
"""
y_pred = clf.predict(val_x)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(val_y, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

yTrain2 = np_utils.to_categorical(trn_y2, 3)
yVal2 = np_utils.to_categorical(val_y2, 3)
input_img = Input(shape=(11,))
layer1 = Dense(6, activation='relu')(input_img)
layer2 = Dense(6, activation='sigmoid')(input_img)
outputLayer = Dense(3, init='normal',activation='softmax')(layer2)
nnModel = Model(input = input_img,output=outputLayer)
nnModel.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
nnModel.fit(trn_x2, yTrain2, batch_size=50, nb_epoch=100,
                  verbose=1, validation_data=(val_x2, yVal2))
"""