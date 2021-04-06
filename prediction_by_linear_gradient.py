# -*- coding: utf-8 -*-


import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

path = '/data.txt'

data2 = pd.read_csv(path, header=None, names=['Size', 'bedrooms', 'Price'])

print('data description')
print(data2.describe())

# rescaling data
data2 = (data2 - data2.mean())/data2.std()

print(data2.head(10))

data2.insert(0, 'Ones', 1)
print(data2.head(10))

cols = data2.shape[1]
X2 = data2.iloc[:, 0:cols-1]
y2 = data2.iloc[:,cols-1:cols]

X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)



theta2 = np.matrix(np.array([0,0,0]))

# intialize learning rates and iterations
alpha = 0.2
iters = 100

# linear regression part 
# 1-- Cost
# 2-- gradient descent function
def computeCost(X, y, theta):
  z = np.power(((X * theta.T) - y), 2)
#    print('z \n',z)
#    print('m ' ,len(X))
  return np.sum(z) / (2 * len(X))




def gradientDescent(x, y, theta, alpha, iters):
  temp = np.matrix(np.zeros(theta.shape))
  parameters = int(theta.ravel().shape[1])
  cost = np.zeros(iters)

  for i in range(iters):
    error = (x * theta.T) - y

    for j in range(parameters):
      term = np.multiply(error, x[:,j])
      temp[0, j] = theta[0, j] - ((alpha / len(x)) * np.sum(term))

    theta = temp
    cost[i] = computeCost(x, y, theta)
  return theta, cost

g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)

cost = computeCost(X2, y2, g2)

# bfl(best fit line)
x = np.linspace(data2.Size.min(), data2.Size.max(), 100)

# linear equation
f = g2[0, 0] + (g2[0, 1] * x)

# line Price vs. Size
fig , ax = plt.subplots(figsize=(5, 5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data2.Size, data2.Price, label='Training Data')
ax.legend(loc = 2)
ax.set_xlabel('Size')
ax.set_ylabel('Price')
ax.set_title('Size vs. Price')

# draw error graph

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')

