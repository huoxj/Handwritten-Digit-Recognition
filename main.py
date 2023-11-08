import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mnist
import nnModule as nm
import network as nn
import msvcrt

train_img, train_label, test_img, test_label = None, None, None, None
def Load_mnist():
    global train_img, train_label, test_img, test_label
    (train_img, train_label), (test_img, test_label) = mnist.load_mnist(one_hot_label = True)

def Training(network, generation):
    batchSize = nn.HyperParam.batchSize
    genePerEpoch = int(mnist.train_num / nn.HyperParam.batchSize)
    plt.ion()
    genePlot, lossPlot= [], []
    plotPerEpoch = 2
    for i in range(0, generation):
        #pick a minibatch
        seed1 = int(time.time())
        np.random.seed(seed1)
        np.random.shuffle(train_img)
        np.random.seed(seed1)
        np.random.shuffle(train_label)
        input = np.split(train_img, [batchSize])[0].T
        input_label = np.split(train_label, [batchSize])[0].T
        #BP
        network.BackPropagate(input, input_label)
        #plotting
        loss = network.GetLoss(input_label)
        genePlot.append(i)
        lossPlot.append(loss)
        plt.ion()
        plt.clf()
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Loss Graph")
        plt.ylim(0, 2.5)
        plt.grid()
        plt.plot(genePlot, lossPlot)
        plt.text(i,min(loss,2.4),"Loss:{:.2f}".format(loss))
        plt.pause(0.001)
        plt.ioff()

def Testing(network):
    testCount = mnist.test_num
    accCount = 0
    for i in range(0, int(testCount)):
        result = network.ForwardPropagate(test_img[i].T)
        if np.argmax(result) == np.argmax(test_label[i]):
            accCount += 1
    return accCount / testCount * 100

if __name__ == "__main__":
    #initialization
    mnist.init_mnist()
    Load_mnist()
    network = nn.Network(np.array([28*28, 64, 32, 32, 32, 10]), nm.ReLU, nm.SGD)
    print("===============mnist load finished===============")
    epoch = int(input("input epoch = "))
    generation = int(epoch * mnist.train_num / nn.HyperParam.batchSize)
    print("generation will be {}".format(generation))
    #do training
    timeStart = time.time()
    Training(network, generation)
    timeEnd = time.time()
    timeTrain = timeEnd - timeStart
    epochPerSec = epoch / timeTrain
    #test network
    timeStart = time.time()
    testAcc = Testing(network)
    timeEnd = time.time()
    timeTest = timeEnd - timeStart
    print("train time: {0:.2f}s\nepoch time : {1:.2f}s\ntest time: {2:.2f}s".format(timeTrain, 1 / epochPerSec, timeTest))
    print("Accuracy : {0:.3f} % \npress any key to quit".format(testAcc))
    msvcrt.getch()