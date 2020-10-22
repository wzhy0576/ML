# Chapter 1 : Linear Regression

### 符号说明：

$$
m=样本数量
$$

$$
n=特征数量
$$

$$
X=\begin{bmatrix}x_0^0 & x_1^0 &... & x_n^0\\ x_0^1 & x_1^1 &... & x_n^1\\ . \\ . \\ x_0^{m-1} & x_1^{m-1} &... & x_n^{m-1}\end{bmatrix}=\begin{bmatrix}x_0 \\ x_1 \\.\\.\\x_{m-1}\end{bmatrix}
$$

$$
y=\begin{bmatrix}y_0 \\ y_1 \\.\\.\\y_{m-1}\end{bmatrix}
$$

$$
\theta^T=\begin{bmatrix}\theta_0 & \theta_1 & ... & \theta_n\end{bmatrix}
$$



### 假设函数：

$$
h_\theta(x)=\theta_{0}x_0+\theta_{1}x_1+\theta_{2}x_2+...+\theta_{n}x_n
$$



### 代价函数：

$$
J(\theta)=\frac1{2m}\sum_{i=0}^{m-1}(h_\theta(x^i)-y^i)^2
$$

```python
J = lambda X, y, theta : np.sum(np.power((X * theta.T - y), 2)) / (2 * len(X))
```



### 梯度下降/迭代方式：

$$
\theta_j=\theta_j-\alpha\frac1m\sum_{i=0}^{m-1}(h_\theta(x^i)-y^i)x^i_j
$$

$$
\theta=\theta-(X\theta^T-y)^TX
$$

```python
new_theta = lambda X, y, alpha, theta : theta - alpha * (X*theta.T - y).T * X / len(X)
```

循环调用new_theta（），直至 J 最小。



### 正规方程组：

$$
\theta=(X^TX)^{-1}X^Ty
$$

```python
theta = lambda X, y : np.linalg.inv(X.T * X) * X.T * y
```



### some tricks:

- 特征放缩

  当特征值范围过大或过小时，采取线性变换，使其分布在大约[-3,3]范围内。

  example:
  $$
  X=\begin{bmatrix}1 & 2.2 & 100\\ 1 & 1.4 & 500\\ 1 & -1.9 & -1000\end{bmatrix}
  $$
  第三列取值范围 1500=500-(-1000) 太大，会拖慢迭代速度。可以将第三列变换成
  $$
  X=\begin{bmatrix}1 & 2.2 & \frac{100}{1500}\\ 1 & 1.4 & \frac{500}{1500}\\ 1 & -1.9 & \frac{-1000}{1500}\end{bmatrix}
  $$
  一般可采用的方所方法有：
  $$
  x=\frac{x}{max - min}
  $$
  
  $$
  \\x=\frac{x-average}{max-min}
  $$



- 多项式回归

  example:
  $$
  x=\begin{bmatrix}x_0 & x_1 & x_2\end{bmatrix}
  $$

  $$
  x^{'}=\begin{bmatrix}x_0 & x_1 & x_2 & x_1^2 & x_2^2 & x_1x_2\end{bmatrix}
  $$

  $$
  h_\theta(x^{'})=\theta_0x_0+\theta_1x_1+\theta_2x_2+\theta_3x_1^2+\theta_4x_2^2+\theta_5x_1x_2
  $$

  

