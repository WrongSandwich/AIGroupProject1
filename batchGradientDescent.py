import numpy as np

X = np.array([[0, 1],[1, 1],[2,1]])
Y = np.array([-1,2,2])

def CFgradient(X, Y, theta):
  j = X.shape[1] #numper of coefficients/regressors
  m = X.shape[0] #number of points
  gradient = np.zeros(j)

  for j in range(j):
    sumTotal = 0
    for i in range(m):
      sumTotal = sumTotal + (X[i,:]*theta - Y[i])*X[i,j]
    gradient[j] = sumTotal/m

  return gradient

theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),np.transpose(X)),Y)

print(theta)
print(CFgradient(X,Y,theta))
