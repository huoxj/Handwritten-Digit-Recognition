import time
import numpy as np
import matplotlib.pyplot as plt
import SourceCode.dataset.mnist as mnist
import nnMath as nm
import nn
import msvcrt

train_img = None
train_label = None
test_img = None
test_label = None
def Load_mnist():
    global train_img,train_label,test_img,test_label
    (train_img, train_label), (test_img, test_label) = mnist.load_mnist(one_hot_label = True)


def TrainNetwork(network, generation):
    batchSize = nn.HyperParam.batchSize
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Graph")
    plt.ylim(0, 2.4)
    plt.ion()
    genePlot = []
    lossPlot = []
    for i in range(0,generation):
        #pickup a minibatch
        seed1 = int(time.time())
        np.random.seed(seed1)
        np.random.shuffle(train_img)
        np.random.seed(seed1)
        np.random.shuffle(train_label)
        input = np.split(train_img, [batchSize])[0].T
        input_label = np.split(train_label, [batchSize])[0].T
        #BP
        network.BackwardPropagate(input, input_label)
        loss = network.GetLoss(input_label)
        #draw loss pic
        genePlot.append(i)
        lossPlot.append(loss)
        plt.clf()
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Loss Graph")
        plt.ylim(0, 2.4)
        plt.grid()
        plt.plot(genePlot, lossPlot)
        plt.text(i,min(loss,2.4),"Loss:{:.2f}".format(loss))
        plt.pause(0.001)
    plt.ioff()


def TestNetwork(network):
    testCount = mnist.test_num
    accCount = 0
    loss = 0
    for i in range(0,int(testCount)):
        result = network.ForwardPropagate(test_img[i].T)
        loss += network.GetLoss(test_label[i])
        if result == np.argmax(test_label[i]):
            accCount += 1
    return accCount / testCount * 100, loss / mnist.test_num

if __name__ == "__main__":
    #initialization
    mnist.init_mnist()
    Load_mnist()
    network = nn.Network(np.array([28*28, 512, 128, 10]), nm.LeakyReLU, nm.dirv_LeakyReLU)
    print("===============mnist load finished===============")
    epoch = int(input("input epoch = "))
    generation = int(epoch * mnist.train_num / nn.HyperParam.batchSize)
    print("generation will be {}".format(generation))
    #do training
    timeStart = time.time()
    TrainNetwork(network, generation)
    timeEnd = time.time()
    timeTrain = timeEnd - timeStart
    epochPerSec = epoch / timeTrain
    #test network
    timeStart = time.time()
    testAcc, testLoss = TestNetwork(network)
    timeEnd = time.time()
    timeTest = timeEnd - timeStart
    print("train time: {0}s   epoch/s : {1}   test time: {2}s".format(timeTrain, epochPerSec, timeTest))
    print("Testset Accuracy : {0}% \npress any key to quit".format(testAcc))
    msvcrt.getch()

#todo:
#   $double loss