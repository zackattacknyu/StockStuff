import xgboost as xgb
import scipy.io as sio
import numpy as np

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