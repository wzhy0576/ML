# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 22:06:20 2020

@author: wzhy

ML Andrew-Ng ex1 version1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

def getData(data_frame):            #DataFrame to Matrix
    data_frame.insert(0, 'Ones', 1)    #insert in col 0 name Ones value 1
    cols = data_frame.shape[1]         #列数
    X = data_frame.iloc[:,0:cols-1]    #取前cols-1列，即输入向量
    y = data_frame.iloc[:,cols-1:cols] #取最后一列，即目标向量
    X = np.matrix(X.values)
    y = np.matrix(y.values)
    return [X, y]

J = lambda X, y, theta : np.sum(np.power((X * theta.T - y), 2)) / (2 * len(X))

new_theta = lambda X, y, alpha, theta : theta - alpha * (X*theta.T - y).T * X / len(X)
    
def gradientDescent(X, y, alpha, times):
    theta = np.matrix([0, 0])
    #cost = np.zeros(times)
    for i in range(times):
        #cost[i] = J(X, y, theta)
        theta = new_theta(X, y, alpha, theta)
    return theta
    
def show(data_frame, final_theta):
    x = np.linspace(data_frame.Population.min(), data_frame.Population.max(), 100)  # 横坐标
    f = final_theta[0, 0] + (final_theta[0, 1] * x)  
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x, f, 'r', label='Prediction')
    ax.scatter(data_frame['Population'], data_frame.Profit, label='Traning Data')
    ax.legend(loc=2)  # 2表示在左上角
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')
    plt.show()
    
path = "./ex1data1.txt"
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
[X, y] = getData(data)
final_theta = gradientDescent(X, y, 0.01, 1000)
show(data, final_theta)
print(final_theta)

LR = linear_model.LinearRegression()
LR.fit(X,y)
print(LR.intercept_)
print(LR.coef_)

# 正规方程
def normalEqn(X, y):
    theta = np.linalg.inv(X.T*X)*X.T*y   #X.T@X等价于X.T.dot(X)
    return theta

final_theta2=normalEqn(X, y)#感觉和批量梯度下降的theta的值有点差距
print(final_theta2)



