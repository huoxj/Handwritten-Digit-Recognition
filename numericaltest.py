import numpy as np
import packages.AutoDiffrentiation as ad

class NumericalTest:
    def __init__(self, graph:ad.graph, test_node:ad.Node):
        graph.compute()
        #get ad result
        self.ad_value = graph.output.value
        self.ad_grad = graph.gradient([test_node])[test_node]
        #get numerical result
        epsilon = 1e-5
        x = test_node.value
        if isinstance(x, np.ndarray):
            nd_grad = np.zeros_like(x)
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    #get dx
                    dx = np.zeros_like(x)
                    dx[i][j] = epsilon
                    #graph clac
                    test_node.set_value(x + dx)
                    graph.compute()
                    hr = graph.output.value
                    test_node.set_value(x - dx)
                    graph.compute()
                    hl = graph.output.value
                    #update grad
                    nd_grad[i][j] = (hr - hl) / (2 * epsilon)
        else:
            dx = epsilon
            test_node.set_value(x + dx)
            graph.compute()
            hr = graph.output.value
            test_node.set_value(x - dx)
            graph.compute()
            hl = graph.output.value
            nd_grad = (hr - hl) / (2 * epsilon)
        self.nd_grad = nd_grad
    def message(self, isPrintLog):
        if isPrintLog:
            print("Numerical diff:")
            print(self.nd_grad)
            print("Auto diff:")
            print(self.ad_grad)
        print("max delta:")
        if isinstance(self.nd_grad, np.ndarray):
            print(max(np.max(self.ad_grad / self.nd_grad), np.max(self.nd_grad / self.ad_grad)))
        else:
            print(abs(self.nd_grad - self.ad_grad))

