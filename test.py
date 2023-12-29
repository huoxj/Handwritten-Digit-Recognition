import numpy as np
import packages.AutoDiffrentiation as ad

def sftmx(x):
    maxVal = np.max(x, 0)
    expx = np.exp(x - maxVal)
    sumExpx = np.sum(expx, 0) + 1e-7
    sft = expx / sumExpx
    return sft

def crossent(x, t):
    return -np.sum(t * np.log(x + 1e-7)) / x.shape[1]

X = 0.1 * np.random.randn(2, 5)
T = np.array([[0,1,0,1,0],[1,0,1,0,1]])

#numerical
epsilon = 1e-7
grad_n = np.zeros_like(X)
for i in range(2):
    for j in range(5):
        dX = np.zeros_like(X)
        dX[i][j] = epsilon
        hr = crossent(sftmx(X + dX), T)
        hl = crossent(sftmx(X - dX), T)
        grad_n[i][j] = (hr - hl) / epsilon / 2
print(crossent(sftmx(X), T))
print(grad_n)
print("============================")
#ad
softmax, iden, sumdown = ad.matrix_softmax(), ad.identityOp(), ad.matrix_sumdown()
mexp, mcs, mdiv, ce = ad.matrix_exp(), ad.matrix_columnSum(), ad.matrix_div(), ad.matrix_crossEntropyLoss()
Xnode = iden(X)
Tnode = iden(T)
expx = mexp(Xnode)
se = mcs(expx)
y = ce(mdiv(expx, se), Tnode)
graph = ad.graph(y)
graph.compute()
print(graph.output.value)
print(graph.gradient([Xnode])[Xnode])

Xnode2 = iden(X)
Tnode2 = iden(T)
y2 = ce(softmax(Xnode2), Tnode2)
graph2 = ad.graph(y2)
graph.compute()
print("============================")
print(graph2.output.value)
print(graph2.gradient([Xnode2])[Xnode2])
