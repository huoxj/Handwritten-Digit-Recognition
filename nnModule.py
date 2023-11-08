import numpy as np
from network import HyperParam as HP

#ACTIVATION FUNC
class ActivationFunc:
    def f(x):
        pass
class ReLU(ActivationFunc):
    def f(x):
        y = np.copy(x)
        y[y < 0] = 0
        return y
    def df(x):
        y = np.ones_like(x)
        y[x < 0] = 0
        return y
class Sigmoid(ActivationFunc):
    def f(x):
        return 1/(1+np.exp(-x))
    def df(x):
        sig = Sigmoid.f(x)
        return sig*(1 - sig)
class LeakyReLU(ActivationFunc):
    def f(x):
        a = x < 0
        y = np.copy(x)
        y[a] = 0.01 * x[a]
        return y
    def df(x):
        y = np.ones_like(x)
        y[x < 0] = 0.01
        return y
    
#OTHER FUNC
class OtherFunc:
    def f(x):
        pass
class Softmax(OtherFunc):
    def f(x):
        maxVal = np.max(x, 0)
        expx = np.exp(x - maxVal)
        sumExpx = np.sum(expx, 0) + 1e-7
        return expx / sumExpx
class CrossEntropyError(OtherFunc):
    def f(x, t):
        return -np.sum(t * np.log(x + 1e-7)) / HP.batchSize
#LEARN METHOD
class LearnMethod:
    def __init__(self):
        pass
    def invoke(self):
        pass
class SGD(LearnMethod):
    def invoke(self, x, partial_x):
        lr = HP.learnRate
        return x - lr * partial_x
class Momentum(LearnMethod):
    def __init__(self):
        self.velo = np.array([])
        self.fri = 0.9
    def invoke(self, x, partial_x):
        lr = HP.learnRate
        if self.velo.size == 0:
            self.velo = np.zeros_like(x)
        else:
            self.velo *= self.fri
        self.velo -= lr * (1 - self.fri) * partial_x
        return x + self.velo
class AdaGrad(LearnMethod):
    def __init__(self):
        self.h = np.array([])
    def invoke(self, x, partial_x):
        lr = HP.learnRate
        if self.h.size == 0:
            self.h = np.zeros_like(x)
        self.h += partial_x * partial_x
        return x - lr * partial_x / (np.sqrt(self.h) + 1e-7)
class Adam(LearnMethod):
    def __init__(self):
        self.velo = np.array([])
        self.h = np.array([])
        self.fri = 0.9
    def invoke(self, x, partial_x):
        lr = HP.learnRate
        if self.h.size == 0:
            self.h = np.zeros_like(x)
        if self.velo.size == 0:
            self.velo = np.zeros_like(x)
        else:
            self.velo *= self.fri
        self.h += partial_x * partial_x
        self.velo += (1 - self.fri) * self.h
        return x - lr * partial_x / (np.sqrt(self.velo) + 1e-7)