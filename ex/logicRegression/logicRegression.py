# -*- coding: utf-8 -*-
#ML Andrew-Ng ex1 version1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

def getData(data_frame):               #DataFrame to Matrix
    data_frame.insert(0, 'Ones', 1)    #insert in col 0 name Ones value 1
    cols = data_frame.shape[1]         #列数
    X = data_frame.iloc[:,0:cols-1]    #取前cols-1列，即输入向量
    y = data_frame.iloc[:,cols-1:cols] #取最后一列，即目标向量
    X = np.matrix(X.values)
    y = np.matrix(y.values)
    return [X, y]

g = lambda x : 1 / (1 + np.exp(-x))

# 代价函数
J = lambda X, y, theta : np.sum(np.power((X * theta.T - y), 2)) / (2 * len(X))

# 经过一趟遍历，得到新的theta
new_theta = lambda X, y, alpha, theta : theta - alpha * (g(X*theta.T) - y).T * X / len(X)
    
def gradientDescent(X, y, alpha, times):
    theta = np.matrix([0, 0, 0])
    #cost = np.zeros(times)
    for i in range(times):
        #cost[i] = J(X, y, theta)
        theta = new_theta(X, y, alpha, theta)
    return theta
    

    
path = "./data1.txt"
data = pd.read_csv(path, header=None, names=['exam1', 'exam2', 'admitted'])
[X, y] = getData(data)
final_theta = gradientDescent(X, y, 0.03, 1000)
print(final_theta)

positive = data[data.admitted.isin(['1'])]  # 1
negetive = data[data.admitted.isin(['0'])]  # 0

fig, ax = plt.subplots(figsize=(6,5))
ax.scatter(positive['exam1'], positive['exam2'], c='b', label='admitted')
ax.scatter(negetive['exam1'], negetive['exam2'], s=50, c='r', marker='x', label='Not Admitted')
# 设置图例显示在图的上方
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width , box.height* 0.8])
ax.legend(loc='center left', bbox_to_anchor=(0.2, 1.12),ncol=3)
# 设置横纵坐标名
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()


x1 = np.arange(130, step=0.1)
x2 = -(final_theta[0][0] + x1*final_theta[0][1]) / final_theta[0][2]
fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(positive['exam1'], positive['exam2'], c='b', label='admitted')
ax.scatter(negetive['exam1'], negetive['exam2'], s=50, c='r', marker='x', label='Not Admitted')
ax.plot(x1, x2)
ax.set_xlim(0, 130)
ax.set_ylim(0, 130)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Decision Boundary')
plt.show()



