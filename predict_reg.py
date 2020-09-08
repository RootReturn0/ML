'''
Author: rootReturn0
Date: 2020-09-08 10:53:58
LastEditors: rootReturn0
LastEditTime: 2020-09-08 17:05:51
Description: 
'''
from pandas.core.common import random_state
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import math
import matplotlib.pyplot as plt

import pandas as pd

FEATURE_NUM = 3

data = pd.read_csv('boston.csv')
# house = datasets.load_boston()
# X = house.data
# Y = house.target

X = data.drop(columns='MEDV')
Y = data['MEDV']

models=['lr','SVM','GBDT','xgboost']

def dataPlot():
    nums = len(X.columns)
    columns =3
    rows = math.ceil(nums/columns)
    plt.figure(figsize=(10,12))
    for i in range(nums):
        plt.subplot(rows,columns,i+1)
        plt.plot(X.iloc[:,i],Y,"b+")
        plt.title(label=X.iloc[:,i].name)
    plt.subplots_adjust(hspace=0.8)
    plt.show()

def featurePlot(feature):
    plt.scatter(data[feature],data['MEDV'])
    plt.show()

def preprocessing():
    global data
    data = data.drop(data[(data['MEDV']>=50) & (data['RM']<8)].index)

def bestFeatures(num):
    stand = StandardScaler()
    stand_x = stand.fit_transform(X)
    best = SelectKBest(f_regression, k=num).fit(stand_x,Y)
    # 最相关特征的index
    best_index = best.get_support()
    print(best_index,'\n',X.columns.values)
    best_features = X.columns.values[best_index]
    print(best_features)
    return best_features

def train(features,models=[]):
    x_train, x_test, y_train, y_test = train_test_split(X[features], Y, test_size=0.2, random_state=11)
    print(X.shape)
    print(x_train.shape)
    print(x_test.shape)
    # print(house)
    
    for model in models:
    #     classifier = selectModel(modelname=model)
    #     classifier.fit(x_train, y_train)
        print(model,'\n')
        predicter = selectModel(model)
        predicter.fit(x_train,y_train)
        preds = predicter.predict(x_test)

        performance(y_test, preds, modelname=model)

def selectModel(modelname):
    if modelname == "lr":
        from sklearn.linear_model import LinearRegression
        clf = LinearRegression()

    elif modelname == "GBDT":
        from sklearn.ensemble import GradientBoostingRegressor
        clf = GradientBoostingRegressor()

    elif modelname == "RF":
        from sklearn.ensemble import RandomForestRegressor
        clf = RandomForestRegressor(n_estimators=100)

    elif modelname == "xgboost":
        from xgboost import XGBRegressor
        clf = XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, objective='reg:gamma')    
    elif modelname == "SVM":
        from sklearn.svm import SVR
        clf = SVR()
    else:
        pass
    return clf

def performance(y_true, y_pred, modelname=""):
    # report = prediction_report(y_true, y_pred)
    # print("模型{}预测结果：\n{}".format(modelname,report))
    evs = explained_variance_score(y_true, y_pred)
    mae = mean_absolute_error(y_true,y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mdae = median_absolute_error(y_true,y_pred)
    r2 = r2_score(y_true,y_pred)
    print('explained_variance_score:',evs)
    print('mean_absolute_error:',mae)
    print('mean_squared_error:',mse)
    print('median_absolute_error',mdae)
    print('r2_score',r2)
    print('\n---\n')
    # return report
if __name__ == "__main__":
    # dataPlot()
    # featurePlot('RM')
    preprocessing()
    # featurePlot('RM')
    best_features = bestFeatures(FEATURE_NUM)
    train(best_features,models)