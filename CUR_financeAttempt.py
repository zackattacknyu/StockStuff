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
patFile = "/home/zdestefa/financeProj/TrainTestData1.mat"
curMATcontent = sio.loadmat(patFile)
x = curMATcontent["Xdata"]
yGiven = curMATcontent["Ydata"]
y = np.reshape(yGiven,(499))

trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x, y, random_state=42, stratify=y,
                                                                   test_size=0.20)

clf = xgb.XGBRegressor(max_depth=10,
                       n_estimators=1500,
                       min_child_weight=9,
                       learning_rate=0.05,
                       nthread=8,
                       subsample=0.80,
                       colsample_bytree=0.80,
                       seed=4242)

clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='mae', early_stopping_rounds=100)

y_pred = clf.predict(val_x)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(val_y, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

yTrain = np_utils.to_categorical(trn_y, 2)
yVal = np_utils.to_categorical(val_y, 2)
input_img = Input(shape=(70,))
layer1 = Dense(32, activation='relu')(input_img)
layer2 = Dense(32, activation='sigmoid')(input_img)
outputLayer = Dense(2, init='normal',activation='softmax')(layer2)
nnModel = Model(input = input_img,output=outputLayer)
nnModel.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
nnModel.fit(trn_x, yTrain, batch_size=50, nb_epoch=100,
                  verbose=1, validation_data=(val_x, yVal))