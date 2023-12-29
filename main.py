import numpy as np
import matplotlib.pyplot as plt
import network
import hyperparams as HP
import numericaltest
import optimizer as om
import packages.mnist as mnist
import time

train_img, train_label, test_img, test_label = None, None, None, None
genePlot, lossPlot = [], []
def Load_mnist():
    mnist.init_mnist()
    global train_img, train_label, test_img, test_label
    (train_img, train_label), (test_img, test_label) = mnist.load_mnist(one_hot_label = True)
def draw(gene, loss):
    genePlot.append(gene)
    lossPlot.append(loss)
    plt.ion()
    plt.clf()
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Graph")
    plt.ylim(0, 2.5)
    plt.grid()
    plt.plot(genePlot, lossPlot)
    plt.text(gene,min(loss,2.4),"Loss:{:.2f}".format(loss))
    plt.pause(0.001)
    plt.ioff()
def init():
    net = network.AfflineNetwork([28 * 28, 800, 10], optimizer=om.Adam)
    return net
def train(net, generation):
    genePlot.clear()
    lossPlot.clear()
    for i in range(generation):
        try:
            seed1 = int(time.time())    
            np.random.seed(seed1)
            np.random.shuffle(train_img)
            np.random.seed(seed1)
            np.random.shuffle(train_label)
            input = np.split(train_img, [HP.BATCHSIZE])[0].T
            input_label = np.split(train_label, [HP.BATCHSIZE])[0].T
            net.set_input(input)
            net.set_label(input_label)
            net.compute()
            net.optimize()
            draw(i, net.get_loss())
        except KeyboardInterrupt:
            print("training stopped by ^c")
    print("loss : {}".format(lossPlot[len(lossPlot) - 1]))
def test(net):
    testCount = mnist.test_num
    trainCount = mnist.train_num
    accTestCount, accTrainCount = 0, 0
    for i in range(0, int(testCount)):
        net.set_input(test_img[i].T.reshape(28 * 28, 1))
        net.set_label(test_label[i].reshape(10,1))
        net.compute()
        result = net.get_result()
        if np.argmax(result) == np.argmax(test_label[i]):
            accTestCount += 1
    accuracy = accTestCount / testCount * 100
    print("Acc test : {:.2f}%".format(accuracy))
    for i in range(0, int(trainCount)):
        net.set_input(train_img[i].T.reshape(28 * 28, 1))
        net.set_label(train_label[i].reshape(10,1))
        net.compute()
        result = net.get_result()
        if np.argmax(result) == np.argmax(train_label[i]):
            accTrainCount += 1
    accuracy = accTrainCount / trainCount * 100
    print("Acc train : {:.2f}%".format(accuracy))
def testSet(nets):
    #vote test
    testCount = mnist.test_num
    trainCount = mnist.train_num
    accTestCount, accTrainCount = 0, 0
    for i in range(0, int(testCount)):
        result = None
        for net in nets:
            net.set_input(test_img[i].T.reshape(28 * 28, 1))
            net.set_label(test_label[i].reshape(10,1))
            net.compute()
            if result is None:
                result = net.get_result()
            else:
                result += net.get_result()
        if np.argmax(result) == np.argmax(test_label[i]):
            accTestCount += 1
    accuracy = accTestCount / testCount * 100
    print("Acc test : {:.2f}%".format(accuracy))

def single_net():
    net = init()
    train(net, 150)
    test(net)
def multi_net():
    netCount = 10
    nets = []
    for i in range(netCount):
        net = init()
        nets.append(net)
        print("Current in net {}".format(i))
        train(net, 250)
    testSet(nets)

Load_mnist()
multi_net()