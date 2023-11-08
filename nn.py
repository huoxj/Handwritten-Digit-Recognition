import numpy as np
import nnMath as nm
import matplotlib.pyplot as plt
import SourceCode.dataset.mnist as mnist

class HyperParam:
    learnRate = 0.03
    batchSize = 5000
    paramInitScale = 0.01

class reference:
    def __init__(self) -> None:
        pass

class NeuroLayer(reference):
    def __init__(self, prevLayer: reference, layerSize: int, prevLayerSize: int, batchSize, activFunc, activFunc_diff):
        self.prevLayer = prevLayer
        if prevLayer != None:
            self.prevLayerSize = prevLayer.layerSize
        else:
            self.prevLayerSize = 1
        self.layerSize = layerSize
        self.activFunc = activFunc
        self.activFunc_diff = activFunc_diff
        self.w = HyperParam.paramInitScale *\
                np.random.randn(self.layerSize, self.prevLayerSize)
        self.b = HyperParam.paramInitScale *\
                np.random.randn(self.layerSize, 1)
        self.a = np.zeros((self.layerSize, batchSize))
        self.z = np.zeros((self.layerSize, batchSize))
    def ForwardPropagate(self):
        #print((np.dot(self.w,self.prevLayer.a).shape,self.b.shape))
        dt = np.dot(self.w,self.prevLayer.a)
        if dt.size == self.b.size:
            self.z = dt.reshape(self.b.shape) + self.b
        else:
            self.z = dt + self.b
        self.a = self.activFunc(self.z)
    def BackPropagate(self, t):
        if self.prevLayer == None:
            return
        a, z, w, al, b = self.a, self.z, self.w, self.prevLayer.a, self.b
        lrate = HyperParam.learnRate
        #get partial driv
        J_z = 2 * (a - t) * self.activFunc_diff(z)
        J_w = np.dot(J_z,al.T)
        J_al = np.dot(w.T, J_z)
        J_b = np.sum(J_z,1).T / HyperParam.batchSize
        #update param
        self.w = w - lrate * J_w
        self.b = b - lrate * J_b.reshape(self.b.shape)
        #recursion
        self.prevLayer.BackPropagate(al - lrate * J_al)

class Network:
    batchSize = HyperParam.batchSize
    def __init__(self, networkSize: np.ndarray, activFunc, activFunc_diff):
        self.networkLayerCount = networkSize.size
        self.layers = np.empty(self.networkLayerCount,dtype=NeuroLayer)
        self.activeFunc = activFunc
        self.activeFunc_diff = activFunc_diff
        for index in range(0,self.networkLayerCount):
            prevLayer = None
            if index > 0 :
                prevLayer = self.layers[index - 1]
            self.layers[index] = NeuroLayer(prevLayer, networkSize[index], 0, self.batchSize, self.activeFunc, self.activeFunc_diff)
    def ForwardPropagate(self, input):
        layers = self.layers
        layerCount = layers.size
        outLayer = layers[layerCount - 1]
        layers[0].a = input
        for index in range(1,layerCount):
            layers[index].ForwardPropagate()
        return np.argmax(outLayer.a)
    def BackwardPropagate(self, input, target):
        self.ForwardPropagate(input)
        layers = self.layers
        layerCount = layers.size
        outLayer = layers[layerCount - 1]
        #gradient descent
        J_a = nm.softmax(outLayer.a)
        J_a[np.bool_(target)] -= 1
        outLayer.BackPropagate(outLayer.a - HyperParam.learnRate * J_a)
    def GetLoss(self, label):
        layers = self.layers
        layerCount = layers.size
        outLayer = layers[layerCount - 1]
        #clac softmax and loss
        sftmax = nm.softmax(outLayer.a)
        loss = nm.cross_entropy_error(sftmax,label)
        return loss
