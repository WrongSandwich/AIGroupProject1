import numpy as np

X = np.array([[0, 1],[1, 1],[2,1]])
Y = np.array([-1,2,2])
Xnew = np.array([[3,1], [4,1],[5,1]])
Ynew = np.array([3,4,5])

class batchLSE:
  def LSE(self, X, Y):
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),np.transpose(X)),Y)

  #Constructor
  def __init__(self):
    self.alpha = 0
    self.theta = 0
  
  def __init__(self, X, Y, alpha = 0):
    self.theta = self.LSE(X, Y)
    self.alpha = alpha

  def setLearningRate(self, alpha):
    self.alpha = alpha
    
  def CFgradient(self, X, Y):
    j = X.shape[1] #numper of coefficients/regressors
    m = X.shape[0] #number of points
    gradient = np.zeros(j)

    for j in range(j):
      sumTotal = 0
      for i in range(m):
        sumTotal = sumTotal + (np.matmul(X[i,:],self.theta) - Y[i])*X[i,j]
        gradient[j] = sumTotal/m

    return gradient
  
  def batchUpdate(self, Xnew, Ynew):
    if self.alpha == 0:
      self.theta = self.LSE(Xnew, Ynew)
    else:
      gradient = self.CFgradient(Xnew, Ynew)
      for j in range(gradient.shape[0]):
        self.theta[j] = self.theta[j] - (self.alpha * gradient[j])
      
  def predict(self, Xpred):
    return np.matmul(Xpred, self.theta)

bLSE = batchLSE(X,Y)
print(bLSE.theta)
bLSE.setLearningRate(0.1)
bLSE.batchUpdate(Xnew, Ynew)
print(bLSE.theta)

alpha_base = .1

nmodels = 4
ftrain = 0.7
ftest = 0.2
fcv = 0.1

timeSeries = np.arange(1929, 2019)
nYearsTrained = round(ftrain*timeSeries.shape[0])
nYearsTesting = round(ftest*timeSeries.shape[0])

np.random.shuffle(timeSeries)
timeSeries = np.transpose(np.array([timeSeries]))
(trainSeries, testSeries, CVSeries) = np.split(timeSeries,[nYearsTrained, nYearsTrained+nYearsTesting])

print(np.amin(timeSeries))
print(np.amax(timeSeries))
print(trainSeries.shape)
print(testSeries.shape)
print(CVSeries.shape)
print(CVSeries)
