from sklearn.datasets import make_classification, make_regression, load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

class Optimizer:
    def optimize(learning_rate, W, b, dW, db):
        pass

class SGD(Optimizer):
    def optimize(self, learning_rate, W, b, dW, db):
        W -= learning_rate * dW
        b -= learning_rate * db
        return W, b

class MomentumSGD(Optimizer):
    def __init__(self, gamma = 0.9):
        self.Vw, self.Vb = 0, 0
        self.gamma = gamma
  
    def optimize(self, learning_rate, W, b, dW, db):
        self.Vw, self.Vb = 0, 0

        self.Vw = (self.gamma * self.Vw) + ((1 - self.gamma) * dW)
        W -= learning_rate * self.Vw

        self.Vb = (self.gamma * self.Vb) + ((1 - self.gamma) * db)
        b -= learning_rate * self.Vb
        return W, b

class Adam(Optimizer):
    def __init__(self, beta0 = 0.9, beta1 = 0.999, eps = 10e-8, t = 9):
        self.first_moment0, self.first_moment1 = 0, 0
        self.second_moment0, self.second_moment1 = 0, 0
        self.beta0 = beta0
        self.beta1 = beta1
        self.eps = eps
        self.t = t

    def optimize(self, learning_rate, W, b, dW, db):
        self.first_moment0, self.first_moment1 = 0, 0
        self.second_moment0, self.second_moment1 = 0, 0

    
        self.first_moment0 = self.beta0 * self.first_moment0 + (1 - self.beta0) * dW
        self.first_moment1 = self.beta0 * self.first_moment1 + (1 - self.beta0) * db

        self.second_moment0 = self.beta1 * self.second_moment0 + (1 - self.beta1) * (dW ** 2)
        self.second_moment1 = self.beta1 * self.second_moment1 + (1 - self.beta1) * (db ** 2)

        first_unbias0 = self.first_moment0 / (1 - self.beta0 ** self.t)
        first_unbias1 = self.first_moment1 / (1 - self.beta0 ** self.t)

        second_unbias0 = self.second_moment0 / (1 - self.beta1 ** self.t)
        second_unbias1 = self.second_moment1 / (1 - self.beta1 ** self.t)

        W -= ((learning_rate * first_unbias0) / np.sqrt(second_unbias0 + self.eps))
        b -= ((learning_rate * first_unbias1) / np.sqrt(second_unbias1 + self.eps))

        return W, b

class NeuralFramework:

  def add_layers(self, layers, loss_function, optimizer=SGD(), 
                 learning_rate=0.02, epochs_count=20,
                 classification = True, regression = False):
    self.net = Net(layers) 
    self.loss_func = loss_function
    self.optimizer = optimizer
    self.lr = learning_rate
    self.epochs_count = epochs_count
    self.cls = classification
    self.rgr = regression

    if self.lr == 0.02:
        if isinstance(self.optimizer, MomentumSGD):
            self.lr = 0.2
        elif isinstance(self.optimizer, Adam):
            self.lr = 0.99

    if self.rgr == True:
        self.cls = False


  def train(self, X_train, Y_train):
    x, y = np.array(X_train), np.array(Y_train)
    n_obs = x.shape[0]

    batch_size = int(n_obs / 10)

    xy = np.c_[x.reshape(n_obs, -1), y.reshape(n_obs, 1)]
    rng = np.random.default_rng(seed = None)

    batch_count = int(n_obs // batch_size)

    print('Training model:')
    for epoch in range(self.epochs_count):
      rng.shuffle(xy)

      for batch in range(batch_count):
        start = batch * batch_size
        stop = start + batch_size

        x_batch, y_batch = xy[start:stop, :-1], xy[start:stop, -1:]

        if self.cls == True:
            y_batch = y_batch.astype('int32')

        # forward pass
        pred, loss = forward(x_batch, y_batch, self.net, self.loss_func)

        # backward pass
        backward(loss, self.lr, self.net, self.loss_func, self.optimizer)

      if self.cls == True:
          acc = accuracy(self.net, xy[:, :-1], xy[:, -1:])
          print(f"Epoch {epoch}: acc = {acc}")
      elif self.rgr == True:
          print(f"Epoch {epoch}: loss = {loss}")


  def predict(self, x_test, y_test):
    obs = x_test.shape[0]
    xy = np.c_[x_test.reshape(obs, -1), y_test.reshape(obs, 1)]
    x_test = xy[:, :-1]
    y_test = xy[:, -1:]

    if self.cls == True:
        y_test = y_test.astype('int32')

    preds = []

    for i in range(obs):
      pred, loss = forward(x_test, y_test, self.net, self.loss_func)

    if self.cls == True:
        pred = np.argmax(pred, axis=1)
        acc = (pred==y_test.astype('int32').flatten()).mean()
        return pred, acc

    elif self.rgr == True:
        preds.append(pred.flatten().tolist())
        return  np.array(preds).flatten(), ((np.array(preds).flatten() - y_test.flatten()) ** 2).mean()

class Net:
  def __init__(self, layers):
    self.layers = layers

  def forward(self, x):
    for layer in self.layers:
      x = layer.forward(x)
    return x

  def backward(self, x):
    for layer in self.layers[::-1]:
      x = layer.backward(x)
    return x

  def update(self, learning_rate, optimizer):
    for layer in self.layers:
      if 'update' in layer.__dir__():
        layer.update(learning_rate, optimizer)

class Linear:
    def __init__(self, nin, nout):
        self.W = np.random.normal(0, 1.0 / np.sqrt(nin), (nout, nin))
        self.b = np.zeros((1, nout))

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W.T) + self.b


    def backward(self, dz):
        self.dW = np.dot(dz.T, self.x)
        self.db = dz.sum(axis=0)
        return np.dot(dz, self.W)

    def update(self, learning_rate, optimizer):
        self.W, self.b = optimizer.optimize(learning_rate, self.W, self.b, self.dW, self.db)

class ReLU:
    def forward(self, x):
        self.y = np.maximum(0, x)
        return self.y

    def backward(self,dy):
        return np.multiply(dy, np.int64(self.y > 0))

class Tanh:
    def forward(self, x):
        self.y = np.tanh(x)
        return self.y

    def backward(self, dy):
        return (1.0 - (self.y ** 2)) * dy

class Sigmoid:
    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, dy):
        return dy * self.y * (1 - self.y)

class Softmax:
    def forward(self, z):
        self.z = z
        zmax = z.max(axis=1, keepdims=True)
        expz = np.exp(z - zmax)
        Z = expz.sum(axis=1, keepdims=True)
        return expz / Z

    def backward(self,dp):
        p = self.forward(self.z)
        pdp = p * dp
        return pdp - p * pdp.sum(axis=1, keepdims=True)

class BinaryCrossEntropy:
    def forward(self, y_pred, y_actual):
        self.y_pred = y_pred
        self.y_actual = y_actual
        p_of_y = y_pred[np.arange(len(y_actual)), y_actual]
        return (-(y_actual * np.log(p_of_y) + (1 - y_actual) * np.log(1 - p_of_y))).mean()

    def backward(self, loss):
        dbin = np.zeros_like(self.y_pred)
        dbin[np.arange(len(self.y_actual)), self.y_actual] -= 1.0 / len(self.y_actual)
        return (((1 - dbin) / (1 - self.y_pred)) + (dbin / self.y_pred))

class CrossEntropyLoss:
    def forward(self,p,y):
        self.p = p
        self.y = y
        p_of_y = p[np.arange(len(y)), y]
        log_prob = - np.log(p_of_y)
        return log_prob.mean()

    def backward(self,loss):
        dlog_softmax = np.zeros_like(self.p)
        dlog_softmax[np.arange(len(self.y)), self.y] -= 1.0/len(self.y)
        return dlog_softmax / self.p

class MeanSquaredError:
    def forward(self, y_pred, y_actual):
        self.y_pred = y_pred
        self.y_actual = y_actual
        return ((y_pred.flatten() - y_actual) ** 2).mean()

    def backward(self, loss):
        self.y_actual = self.y_actual.reshape(-1,1)
        return 2 * (self.y_pred - self.y_actual)

def forward(x_batch, y_batch, net, loss_func):
    pred = net.forward(x_batch)
    loss = loss_func.forward(pred, y_batch.flatten())
    return pred, loss

def backward(loss, learning_rate, net, loss_func, optimizer):
    dpred = loss_func.backward(loss)
    grad = net.backward(dpred)

    # update weights
    net.update(learning_rate, optimizer)

def accuracy(net, x, y):
    z = net.forward(x)
    pred = np.argmax(z, axis=1)
    return (pred==y.flatten()).mean()
