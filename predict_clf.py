'''
Author: rootReturn0
Date: 2020-09-08 10:53:58
LastEditors: rootReturn0
LastEditTime: 2020-09-08 17:30:38
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
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression()

    elif modelname == "GBDT":
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier()

    elif modelname == "RF":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100)

    elif modelname == "xgboost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(
                learning_rate=0.01,
                n_estimators=1000,
                max_depth=4,
                min_child_weight=3,
                gamma=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1,
                objective='binary:logistic', #multi:softmax
                nthread=8,
                scale_pos_weight=1,
                seed=27,
                random_state=27
            )    
    elif modelname == "KNN":
        from sklearn.neighbors import KNeighborsClassifier as knn
        clf = knn()
    
    elif modelname == "MNB":
        from sklearn.naive_bayes import MultinomialNB
        clf = MultinomialNB()

    elif modelname == "SVM-P":
        from sklearn.svm import SVC
        clf = SVC(kernel = 'poly', probability = True)

    elif modelname == "SVM-R":
        from sklearn.svm import SVC
        clf = SVC(kernel = 'rbf', probability = True)
    else:
        pass
    return clf

def performance(y_true, y_pred, modelname=""):
    report = classification_report(y_true, y_pred)
    print("模型{}预测结果：\n{}".format(modelname,report))
    return report
if __name__ == "__main__":
    # dataPlot()
    # featurePlot('RM')
    preprocessing()
    # featurePlot('RM')
    best_features = bestFeatures(FEATURE_NUM)
    train(best_features,models)