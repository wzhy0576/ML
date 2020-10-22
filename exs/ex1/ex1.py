# -*- coding: utf-8 -*-
"""
Spyder Editor

ML Andrew-Ng ex1 version0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "ex1data1.txt"   #ex1data1.txt should be in current directory
dataset = pd.read_csv(path, header=None, names=['Population', 'Profit'])
dataset.plot(kind='scatter', x='Population', y='Profit', figsize=(8,5))
plt.show()

def Cost(X, y, theta):
    inner = np.power((X * theta.T - y), 2)
    print(inner)
    return np.sum(inner) / (2 * len(X))

dataset.insert(0, 'Ones', 1)    #insert in col 0 name Ones value 1
# set X (training data) and y (target variable)
cols = dataset.shape[1]  # 列数
X = dataset.iloc[:,0:cols-1]  # 取前cols-1列，即输入向量
y = dataset.iloc[:,cols-1:cols] # 取最后一列，即目标向量
X = np.matrix(X.values)
y = np.matrix(y.values)

theta = np.matrix([0,0])
np.array([[0,0]]).shape 

#print(X.shape, theta.shape, y.shape)
#print(Cost(X, y, theta))

def gradientDescent(X, y, theta, alpha, epoch):
    """reuturn theta, cost"""
        
    temp = np.matrix(np.zeros(theta.shape))  # 初始化一个 θ 临时矩阵(1, 2)
    #parameters = int(theta.flatten().shape[1])  # 参数 θ的数量
    cost = np.zeros(epoch)  # 初始化一个ndarray，包含每次epoch的cost
    m = X.shape[0]  # 样本数量m
    
    for i in range(epoch):
        # 利用向量化一步求解
        temp =theta - (alpha / m) * (X * theta.T - y).T * X
        theta = temp
        cost[i] = Cost(X, y, theta) 
    return theta, cost

alpha = 0.01
epoch = 1000

final_theta, cost = gradientDescent(X, y, theta, alpha, epoch)

Cost(X, y, final_theta)

x = np.linspace(dataset.Population.min(), dataset.Population.max(), 100)  # 横坐标
f = final_theta[0, 0] + (final_theta[0, 1] * x)  # 纵坐标，利润

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(dataset['Population'], dataset.Profit, label='Traning Data')
ax.legend(loc=2)  # 2表示在左上角
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(np.arange(epoch), cost, 'r')  # np.arange()返回等差数组
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()




