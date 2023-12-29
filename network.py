import numpy as np
import layers as nnl
import packages.AutoDiffrentiation as ad
import optimizer as om
import hyperparams as HP

class Network:
    def __init__(self, layers:list, activFunc = ad.matrix_ReLU(), optimizer = om.SGD):
        pass
    def set_input(self, inputmatrix):
        pass

class AfflineNetwork(Network):
    def __init__(self, layersizes: list, activFunc=ad.matrix_ReLU(), optimizer=om.SGD, initScaleMult = 2):
        self.layers = []
        layer_input = nnl.layer_input(np.zeros((layersizes[0], HP.BATCHSIZE)));
        self.layers.append(layer_input)
        layer_last = layer_input
        for i in range(1, len(layersizes)):
            layer_cur = nnl.layer_affline(layersizes[i], initScaleMult, layer_last, activFunc, optimizer)
            self.layers.append(layer_cur)
            layer_last = layer_cur
        layer_output = nnl.layer_softmax(layer_last)
        self.layers.append(layer_output)
        self.graph = ad.graph(layer_output.get_output())
    def compute(self):
        self.graph.compute()
    def optimize(self):
        self.graph.optimize()
    def set_input(self, inputmatrix):
        self.layers[0].set_input(inputmatrix)
    def get_result(self):
        return self.layers[len(self.layers) - 1].get_result()
    def set_label(self, label):
        layercount = len(self.layers)
        self.layers[layercount - 1].set_label(label)
    def get_loss(self):
        layercount = len(self.layers)
        return self.layers[layercount - 1].get_loss()
    def get_layers(self):
        return self.layers

class AfflineNetwork_batchnorm(Network):
    def __init__(self, layersizes: list, activFunc=ad.matrix_ReLU(), optimizer=om.SGD, initScaleMult = 2):
        self.layers = []
        layer_input = nnl.layer_input(np.zeros((layersizes[0], HP.BATCHSIZE)));
        self.layers.append(layer_input)
        layer_last = layer_input
        for i in range(1, len(layersizes)):
            layer_cur = nnl.layer_affline(layersizes[i], initScaleMult, layer_last, activFunc, optimizer)
            self.layers.append(layer_cur)
            layer_bat = nnl.layer_batchNorm(layer_cur, optimizer)
            self.layers.append(layer_bat)
            layer_last = layer_bat
        layer_output = nnl.layer_softmax(layer_last)
        self.layers.append(layer_output)
        self.graph = ad.graph(layer_output.get_output())
    def compute(self):
        self.graph.compute()
    def optimize(self):
        self.graph.optimize()
    def set_input(self, inputmatrix):
        self.layers[0].set_input(inputmatrix)
    def get_result(self):
        return self.layers[len(self.layers) - 1].get_result()
    def set_label(self, label):
        layercount = len(self.layers)
        self.layers[layercount - 1].set_label(label)
    def get_loss(self):
        layercount = len(self.layers)
        return self.layers[layercount - 1].get_loss()
    def get_layers(self):
        return self.layers
