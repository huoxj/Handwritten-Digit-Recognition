import numpy as np
from math import exp
import nn

def sigmoid(x):
    return 1/(1+np.exp(-x))
def dirv_sigmoid(x):
    sig = sigmoid(x)
    return sig*(1 - sig)

def ReLU(x):
    a = x < 0
    y = np.copy(x)
    y[a] = 0
    return y
def dirv_ReLU(x):
    a = x > 0
    y = np.zeros_like(x)
    y[a] = 1
    return y

def LeakyReLU(x):
    a = x < 0
    y = np.copy(x)
    y[a] = 0.01 * x[a]
    return y
def dirv_LeakyReLU(x):
    a = x < 0
    y = np.ones_like(x)
    y[a] = 0.01
    return y

def softmax(x):
    maxNum = np.max(x, 0)
    expx = np.exp(x - maxNum)
    sumExpx = np.sum(expx, 0) + 1e-7
    y = expx / sumExpx
    return y

def cross_entropy_error(x, t):
    deltax = 1e-7
    return -np.sum(t * np.log(x + deltax)) / nn.HyperParam.batchSize