import numpy as np
import nnModule as nm

class temp:
    def __init__(self, word):
        self.word = word
    def a(self):
        print(self.word)

def b(method):
    m1 = method("hello1")
    m2 = method("hello2")
    m1.a()
    m2.a()

x = np.array([[1,2],[3,4]],dtype=float)
px = np.array([[0.1,0.2],[0.3,0.4]])
lm = nm.Momentum()
x = lm.invoke(x, px)
print(x)