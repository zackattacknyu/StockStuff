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
patFile = "/home/zdestefa/financeProj/TrainTestData7.mat"
curMATcontent = sio.loadmat(patFile)
trn_x = curMATcontent["Xtrain"]
val_x = curMATcontent["Xtest"]
trn_y = curMATcontent["Ytrain"]
val_y = curMATcontent["Ytest"]

trn_y = np.reshape(trn_y,(448))
val_y = np.reshape(val_y,(50))

clf = xgb.XGBRegressor(max_depth=10, n_estimators=300, min_child_weight=9,learning_rate=0.05,nthread=8,subsample=0.80,colsample_bytree=0.80,seed=4242)

clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='rmse', early_stopping_rounds=100)

yHatTest = clf.predict(val_x)

sio.savemat('stockResults7.mat',{'yHatTest':yHatTest})