import numpy as np
import pandas as pd

class LinearModel :
    # 实现：基于带标签的数据集，训练得出线性预测器
    # 注意：本类依赖于numpy库，若使用本类，请"import numpy as np"
    # 实现的预测器在样本数量小于10000时使用正规方程，否则使用梯度下降
    # 并且本类的梯度下降不需要手动指定学习率和迭代次数
    
    fitSuccess = False  #初始化为未拟合状态
    
    def setData(self, dataFrame):               
        # 设置用于训练的数据集
        # 使用pandas.read_scsv()获取合适的dataFrame
        # a line in your dataFrame should be (x1, x2, ...., xn, y)        
        dataFrame.insert(0, 'Ones', 1)          #最左边插入一列全1，便于矩阵统一运算
        cols = dataFrame.shape[1]               #列数
        X = dataFrame.iloc[:,0:cols-1]          #取前cols-1列，即输入矩阵
        y = dataFrame.iloc[:,cols-1:cols]       #取最后一列，即标签向量
        self.X = np.matrix(X.values)
        self.y = np.matrix(y.values)
        self.numSamples = len(X)                #设置样本数
        self.numFeatures = self.X.shape[1] - 1  #设置特征数
       
    def fit(self):
        if(self.numSamples < 10000):    
            #样本数较小，使用“正规方程”得到模型参数
            self.fitWithFormalEquations()
        else:       
            #样本数较大，采用“梯度下降”迭代得到模型参数
            self.fitWithGraddientDescent()
        
    def fitWithFormalEquations(self):
        self.theta = (np.linalg.inv(self.X.T * self.X) * self.X.T * self.y).T
        self.fitSuccess = True
    
    def fitWithGraddientDescent(self):
        self.theta = np.matrix(np.zeros(self.numFeatures + 1))
        self.cost = self.getCost()
        maxTimes = 9999999999   #循环上限，防止死循环
        self.alpha = 0.01   #超参：学习率
        self.cost = 0
        while(maxTimes > 0 and abs(self.cost - self.getCost()) > 0.000000000001):   
            #认为前后两次cost之差小于等于0.000000000001时达到最优状态
            if(self.cost < self.getCost()): 
                #若误差增大，说明学习率太大，在最优解两侧来回晃，应减小alpha
                self.alpha = self.alpha / 3
            elif(self.cost > self.getCost() and abs(self.cost - self.getCost()) / self.cost < 0.1):
                #若前后两次误差变化不明显，说明学习率太小，为加快学习速度，应增大alpha
                self.alpha = self.alpha * 3
            self.cost = self.getCost()
            self.theta = self.theta - self.alpha * self.getGradient()          
            maxTimes = maxTimes - 1    
        if(maxTimes == 0):
            self.fitSuccess = False
        else:
            self.fitSuccess = True
    
    def getCost(self):
        return np.sum(np.power((self.X * self.theta.T - self.y), 2)) / (2 * self.numSamples)
    
    def getGradient(self):
        return (self.X * self.theta.T - self.y).T * self.X / self.numSamples
    
    def predict(self, X):
        if(self.fitSuccess == True and X.shape[0] == 1 and X.shape[1] == self.numFeatures):
            return self.predictOne(X)
        #elif(self.fitSuccess == True and X.shape[0] > 1 and X.shape[1] == self.numFeatures):
            #return self.predictSome(X)
        else:
            return None
        
    def predictOne(self, x):
        theta0 = np.array(self.theta)[0][0]
        theta = np.matrix(np.array(self.theta)[0][1:])
        return theta.T * x + theta0
   

# example     
lm = LinearModel()
path = "./data1.txt"
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

lm.setData(data)
lm.fit()
print(lm.theta)
x = np.matrix([20])
print(lm.predict(x))


