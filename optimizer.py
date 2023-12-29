import numpy as np
import math
from hyperparams import LEARN_RATE as lrate

class Optimizer:
    def __init__(self):
        pass
    def run(self, opt_matrix, graident):
        return opt_matrix

"""
==========================subclass optimizers==========================
"""

class SGD(Optimizer):
    def run(self, optrand, gradient):
        return optrand - lrate * gradient

class Momentum(Optimizer):
    momentum = 0.9
    history = None
    def run(self, optrand, gradient):
        if self.history is None:
            self.history = np.zeros_like(gradient)
        self.history = self.history * self.momentum + lrate * gradient
        return optrand - self.history
    
class AdaGrad(Optimizer):
    history = None
    def run(self, optrand, gradient):
        if self.history is None:
            self.history = np.zeros_like(gradient)
        self.history += gradient * gradient
        return optrand - lrate * gradient / np.sqrt(self.history + 1e-7)

class myAdaGrad(Optimizer):
    history = None
    momentum = 0.9
    momentumaParam = 0
    def run(self, optrand, gradient):
        if self.history is None:
            self.history = np.zeros_like(gradient)
        self.history = self.history * self.get_momentum() + gradient * gradient
        return optrand - lrate * gradient / np.sqrt(self.history + 1e-7)
    def get_momentum(self):
        thrust = self.momentumaParam * 1.0 / 300
        sigmoid = 0.2 / (1 + math.exp(- thrust)) - 0.1
        self.momentumaParam += 1
        return self.momentum + sigmoid

class myAdam(Optimizer):
    history1 = None
    history2 = None
    momentum1 = 0.9
    momentum2 = 0.9
    momentum2Param = 0
    def run(self, optrand, gradient):
        if self.history1 is None:
            self.history1 = np.zeros_like(gradient)
        if self.history2 is None:
            self.history2 = np.zeros_like(gradient)
        self.history1 = self.history1 * self.momentum1 + gradient
        self.history2 = self.history2 * self.get_momentum() + gradient * gradient
        return optrand - lrate * self.history1 / np.sqrt(self.history2 + 1e-7)
    def get_momentum(self):
        thrust = self.momentum2Param * 1.0 / 200
        sigmoid = 0.2 / (1 + math.exp(- thrust)) - 0.1
        self.momentum2Param += 1
        return self.momentum2 + sigmoid

class Adam(Optimizer):
    history1 = None
    history2 = None
    momentum1 = 0.9
    momentum2 = 0.999
    momentum2Param = 0
    def run(self, optrand, gradient):
        if self.history1 is None:
            self.history1 = np.zeros_like(gradient)
        if self.history2 is None:
            self.history2 = np.zeros_like(gradient)
        self.history1 = self.history1 * self.momentum1 + gradient * (1 - self.momentum1)
        self.history2 = self.history2 * self.momentum2 + gradient * gradient * (1 - self.momentum2)
        return optrand - lrate * self.history1 / np.sqrt(self.history2 + 1e-7)