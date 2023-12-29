import packages.AutoDiffrentiation as ad
import numpy as np
import math

iden, mdot, madd, maddcv = ad.identityOp(), ad.matrix_dot(), ad.matrix_add(), ad.matrix_add_cvector()
class layer:
    def __init__(self, layerSize):
        self.layerSize = layerSize
        self.output = None
    def get_output(self):
        return self.output

"""
=======================subclass layers=======================
"""

class layer_input(layer):
    """
    layer with input function
    process: none
    output: input matrix
    """
    def __init__(self, inputMatrix):
        self.output = iden(inputMatrix)
        self.layerSize = inputMatrix.shape[0]
        self.nextLayer = None
    def set_input(self, newInputMatrix):
        self.output.set_value(newInputMatrix)


class layer_affline(layer):
    """
    affline middle layer
    process: linear trans + activFunc
    input, output
    """
    def __init__(self, layerSize, initScaleMult, prevLayer, activFunc, optimizer):
        super().__init__(layerSize)
        #deal with input
        self.prevLayer = prevLayer
        self.input = prevLayer.output
        #init params
        prevSize = prevLayer.layerSize
        initScale = math.sqrt(initScaleMult * 1.0 / prevSize)
        self.W = iden(initScale * np.random.randn(layerSize, prevSize))
        self.B = iden(initScale * np.random.randn(layerSize, 1))
        self.X = self.input
        self.Z = maddcv(mdot(self.W, self.X), self.B)
        self.A = activFunc(self.Z)
        #set up optimizer
        self.W.set_optimizer(optimizer)
        self.B.set_optimizer(optimizer)
        #deal with output
        self.output = self.A


sftmax, crossent = ad.matrix_softmax(), ad.matrix_crossEntropyLoss()
mexp, mcs, mdiv = ad.matrix_exp(), ad.matrix_columnSum(), ad.matrix_div()
class layer_softmax(layer):
    """
    softmax layer with cross entropy loss
    """
    def __init__(self, prevLayer):
        self.prevLayer = prevLayer
        self.input = prevLayer.output
        self.layerSize = self.input.value.shape[0]
        #clac
        self.X = self.input
        expx = mexp(self.X)
        se = mcs(expx)
        self.softmax = mdiv(expx, se)
        self.label = iden(np.zeros_like(self.softmax))
        self.J = crossent(self.softmax, self.label)
        #deal with output
        self.output = self.J
    def set_label(self, label):
        self.label.set_value(label)
    def get_loss(self):
        return self.J.value
    def get_result(self):
        return self.softmax.value


colsum, mulscalar, addscalar, add, neg, div, mul = ad.matrix_columnSum(), ad.matrix_mul_scalar(), ad.matrix_add_scalar(), ad.matrix_add(), ad.matrix_neg(), ad.matrix_div(), ad.matrix_mul()
sqrt = ad.matrix_sqrt()
class layer_batchNorm(layer):
    """
    batch normalization layer
    """
    def __init__(self, prevLayer, optimizer):
        self.prevLayer = prevLayer
        self.input = prevLayer.output
        self.layerSize = self.input.value.shape[0]
        #batch norm
        X = self.input
        mean = mulscalar(colsum(X), iden(1.0 / X.value.shape[1]))  #mean
        meanFix = add(X, neg(mean))
        var = colsum(mul(meanFix, meanFix))
        epsilon = iden(np.zeros_like(var.value) + 1e-7)
        SD = sqrt(add(var, epsilon))
        self.norm = div(meanFix, SD)
        self.scale = iden((float)(1))
        self.bias = iden((float)(0))
        #set up optimizer
        self.scale.set_optimizer(optimizer)
        self.bias.set_optimizer(optimizer)
        #deal with output
        self.output = addscalar(mulscalar(self.norm, self.scale), self.bias)
