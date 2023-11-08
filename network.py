import numpy as np
import nnModule as nm
import math

class HyperParam:
    batchSize = 4000
    learnRate = 0.05
    initScale = 2 / math.sqrt(28 * 28)
class reference:
    def __init__(self) -> None:
        pass

class NeuroLayer(reference):
    def __init__(self, size, prevLayer, activFunc, learnMethod):
        self.size = size
        self.prevLayer = prevLayer
        self.activFunc = activFunc
        self.sizePrev = 1
        if prevLayer != None:
            self.sizePrev = prevLayer.size
        IS = HyperParam.initScale
        self.w = IS * np.random.randn(size, self.sizePrev)
        self.b = IS * np.random.randn(size, 1)
        self.a = None
        self.z = None
        self.upd_w = learnMethod()
        self.upd_b = learnMethod()
        self.upd_al = learnMethod() 
    def ForwardPropagate(self):
        wa = np.dot(self.w, self.prevLayer.a)
        if wa.size == self.b.size:
            wa = wa.reshape(self.b.shape)
        self.z = wa + self.b
        self.a = self.activFunc.f(self.z)
    def BackPropagate(self, t):
        if self.prevLayer == None:
            return
        a, z, w, al, b = self.a, self.z, self.w, self.prevLayer.a, self.b
        #get gradient
        Jz = 2 * (a - t) * self.activFunc.df(z)
        Jw = np.dot(Jz, al.T)
        Jal = np.dot(w.T, Jz)
        Jb = (np.sum(Jz,1).T / HyperParam.batchSize).reshape(b.shape)
        #updateparam
        self.w = self.upd_w.invoke(w, Jw)
        self.b = self.upd_b.invoke(b, Jb)
        self.prevLayer.BackPropagate(self.upd_al.invoke(al, Jal))

class Network:
    def __init__(self, layerSize, activFunc, learnMethod):
        self.size = layerSize.size
        self.activFunc = activFunc
        self.learnMethod = learnMethod
        self.upd_softmax = learnMethod()
        self.layers = np.empty(self.size, dtype = NeuroLayer)
        for i in range(0, self.size):
            prevLayer = self.layers[i - 1]
            self.layers[i] = NeuroLayer(layerSize[i], prevLayer, activFunc, learnMethod)
    def ForwardPropagate(self, input):
        layers = self.layers
        layerCount = self.size
        outLayer = layers[layerCount - 1]
        if input.ndim == 1:
            input = np.reshape(input, (input.size, 1))
        layers[0].a = input
        for i in range(1, layerCount):
            layers[i].ForwardPropagate()
        return outLayer.a
    def BackPropagate(self, input, t):
        if input.ndim == 1:
            input = np.reshape(input, (input.size, 1))
        self.ForwardPropagate(input)
        layers = self.layers
        layerCount = self.size
        outLayer = layers[layerCount - 1]
        #softmax operation
        Ja = nm.Softmax.f(outLayer.a)
        Ja[np.bool_(t)] -= 1
        outLayer.BackPropagate(self.upd_softmax.invoke(outLayer.a, Ja))
    def GetLoss(self, label):
        sftmax = nm.Softmax.f(self.layers[self.size - 1].a)
        loss = nm.CrossEntropyError.f(sftmax, label)
        return loss
