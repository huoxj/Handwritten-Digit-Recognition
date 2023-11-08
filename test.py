import nnMath as nm
import numpy as np
import SourceCode.dataset.mnist as mnist
import time
import matplotlib.pyplot as plt

def dynamic_plot():
    x = []
    y = []
    plt.ion()
    for i in range(0,100):
        y.append(i**2)
        x.append(i)
        plt.clf()
        plt.plot(x,y)
        plt.pause(0.01)
        plt.ioff()
    t = input("press any key to quit")

def little_test():
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Graph")
    y = np.random.randn(10)
    x = np.arange(0,10)
    plt.plot(x,y)
    plt.scatter(x[x.size - 1], y[y.size - 1])
    plt.text(x[x.size - 1], y[y.size - 1], "({},{:.1f})".format(x[x.size - 1], y[y.size - 1]))
    plt.show()
little_test()