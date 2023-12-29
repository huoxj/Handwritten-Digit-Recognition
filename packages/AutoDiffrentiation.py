import numpy as np
import math
class Op:
    def __call__(self):
        """
        创建Node保存计算图节点
        """
        assert False, "请调用子类方法"
    def compute(self, inputs):
        """
        计算节点输出
        """
        assert False, "请调用子类方法"
    def gradient(self, inputs, grad_value):
        """
        计算结点各输入梯度
        """
        assert False, "请调用子类方法"

class Node:
    def __init__(self, op, inputs):
        self.op = op
        self.inputs = inputs
        self.grad = None
        self.optimizer = None
        self.value = 0
        self.outdegree = 0
        self.clac()
        for n in inputs:
            n.outdegree += 1
    def clac(self):
        if isinstance(self.op, identityOp):
            return
        temp_value = self.op.compute(self.inputs)
        self.value = temp_value
    def set_value(self, value):
        if not isinstance(self.op, identityOp):
            assert False, "set_value only valid on identityOp but not " + self.__repr__()
        self.value = value
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer()
    def gradient(self):
        inputs_grad = self.op.gradient(self.inputs, self.grad)
        for n, i in zip(self.inputs, range(len(self.inputs))):
            n.grad = inputs_grad[i] if n.grad is None else n.grad + inputs_grad[i]
    def __repr__(self):
        return self.op.__repr__()
class graph:
    def __init__(self, output):
        self.output = output
        self.topoList = []
        self.topoSort(self.output)
        self.reverseTopoList = self.topoList.copy()
        self.reverseTopoList.reverse()
    def topoSort(self, node):
        for n in node.inputs:
            n.outdegree -= 1
            if n.outdegree == 0:
                self.topoSort(n)
        self.topoList.append(node)
    def compute(self):
        """
        正向传播计算图求值
        return: output_value
        """
        for n in self.topoList:
            n.clac()
            n.grad = None
        return self.output.value
    def optimize(self):
        """
        反向传播更新拥有优化器的参数
        """
        self.gradient([])
        for n in self.topoList:
            if n.optimizer == None:
                continue
            n.value = n.optimizer.run(n.value, n.grad)

    def gradient(self, grad_nodes):
        """
        求对应梯度
        grad_nodes: 要求梯度的节点列表
        root: 对root节点求梯度
        return: 对应梯度字典
        """
        root = self.output
        #计算梯度
        root.grad = np.ones_like(root.value)
        grad_dict = {}
        for n in self.reverseTopoList:
            n.gradient()
            if n in grad_nodes:
                grad_dict[n] = n.grad
        return grad_dict

class identityOp(Op):
    def __call__(self, a):
        node = Node(self, [])
        node.value = a
        return node
    def compute(self, inputs):
        return None
    def gradient(self, inputs, grad_value):
        return []
    def __repr__(self):
        return "Identity"
class add(Op):
    def __call__(self, a, b):
        return Node(self, [a, b])
    def compute(self, inputs):
        assert len(inputs) == 2, "add operation param count invalid(!=2)"
        return inputs[0].value + inputs[1].value
    def gradient(self, inputs, grad_value):
        return [grad_value] * len(inputs)
    def __repr__(self):
        return "+"
class mul(Op):
    def __call__(self, a, b):
        return Node(self, [a, b])
    def compute(self, inputs):
        assert len(inputs) == 2, "multiply operation param count invalid(!=2)"
        return inputs[0].value * inputs[1].value
    def gradient(self, inputs, grad_value):
        return [grad_value * inputs[1].value, grad_value * inputs[0].value]
    def __repr__(self):
        return "*"
class reciprocal(Op):
    def __call__(self, a):
        return Node(self, [a])
    def compute(self, inputs):
        assert len(inputs) == 1, "reciprocal operation param count invalid(!=2)"
        return 1 / inputs[0].value
    def gradient(self, inputs, grad_value):
        return [- grad_value / (inputs[0].value**2)]
    def __repr__(self) -> str:
        return "^-1"
class exp(Op):
    def __call__(self, a):
        return Node(self, [a])
    def compute(self, inputs):
        assert len(inputs) == 1, "exp operation param count invalid(!=2)"
        return math.exp(inputs[0].value)
    def gradient(self, inputs, grad_value):
        return [grad_value * math.exp(inputs[0].value)]
    def __repr__(self) -> str:
        return "exp"
class exp2(Op):
    def __call__(self, a):
        return Node(self, [a])
    def compute(self, inputs):
        assert len(inputs) == 1, "exp2 operation param count invalid(!=2)"
        return math.exp2(inputs[0].value)
    def gradient(self, inputs, grad_value):
        return [grad_value * 2 * inputs[0].value]
    def __repr__(self) -> str:
        return "^2"
class sin(Op):
    def __call__(self, a):
        return Node(self, [a])
    def compute(self, inputs):
        assert len(inputs) == 1, "sin operation param count invalid(!=2)"
        return math.sin(inputs[0].value)
    def gradient(self, inputs, grad_value):
        return [grad_value * math.cos(inputs[0].value)]
    def __repr__(self) -> str:
        return "sin"
class cos(Op):
    def __call__(self, a):
        return Node(self, [a])
    def compute(self, inputs):
        assert len(inputs) == 1, "cos operation param count invalid(!=2)"
        return math.cos(inputs[0].value)
    def gradient(self, inputs, grad_value):
        return [- grad_value * math.sin(inputs[0].value)]
    def __repr__(self) -> str:
        return "cos"

class matrix_add(Op):
    def __call__(self, a, b):
        return Node(self, [a, b])
    def compute(self, inputs):
        return inputs[0].value + inputs[1].value
    def gradient(self, inputs, grad_value):
        return [grad_value] * 2
    def __repr__(self):
        return "m+"
class matrix_add_cvector(Op):
    def __call__(self, a, b):
        return Node(self, [a, b])
    def compute(self, inputs):
        return inputs[0].value + inputs[1].value
    def gradient(self, inputs, grad_value):
        return [grad_value, grad_value.sum(1).reshape(inputs[1].value.shape)]
    def __repr__(self):
        return "m+"
class matrix_add_scalar(Op):
    def __call__(self, a:np.ndarray, b:float):
        return Node(self, [a, b])
    def compute(self, inputs):
        return inputs[0].value + inputs[1].value
    def gradient(self, inputs, grad_value):
        return [grad_value, grad_value.sum()]
    def __repr__(self):
        return "m+"
class matrix_neg(Op):
    def __call__(self, a):
        return Node(self, [a])
    def compute(self, inputs):
        return -inputs[0].value
    def gradient(self, inputs, grad_value):
        return [- grad_value]
    def __repr__(self):
        return "-m"
class matrix_mul_scalar(Op):
    def __call__(self, a:np.ndarray, b:float):
        return Node(self, [a, b])
    def compute(self, inputs):
        return inputs[0].value * inputs[1].value
    def gradient(self, inputs, grad_value):
        return [inputs[1].value * grad_value, 0]
    def __repr__(self):
        return "m*scalar"
class matrix_mul(Op):
    def __call__(self, a, b):
        return Node(self, [a, b])
    def compute(self, inputs):
        return inputs[0].value * inputs[1].value
    def gradient(self, inputs, grad_value):
        return [grad_value * inputs[1].value, grad_value * inputs[0].value]
    def __repr__(self):
        return "m*"
class matrix_dot(Op):
    def __call__(self, a, b):
        return Node(self, [a, b])
    def compute(self, inputs):
        return np.dot(inputs[0].value, inputs[1].value)
    def gradient(self, inputs, grad_value):
        return [np.dot(grad_value, inputs[1].value.T), np.dot(inputs[0].value.T, grad_value)]
    def __repr__(self):
        return "mdot"
class matrix_sumdown(Op):
    def __call__(self, a):
        return Node(self, [a])
    def compute(self, inputs):
        return np.sum(inputs[0].value)
    def gradient(self, inputs, grad_value):
        return [grad_value * np.ones_like(inputs[0].value)]
    def __repr__(self):
        return "sumdown"
class matrix_ReLU(Op):
    def __call__(self, a):
        return Node(self, [a])
    def compute(self, inputs):
        ret = np.copy(inputs[0].value)
        ret[inputs[0].value < 0] = 0
        return ret
    def gradient(self, inputs, grad_value):
        ret = np.copy(grad_value)
        ret[inputs[0].value < 0] = 0;
        return [ret]
    def __repr__(self):
        return "ReLU"
class matrix_softmax(Op):
    def __call__(self, a):
        return Node(self, [a])
    def compute(self, inputs):
        x = inputs[0].value
        maxVal = np.max(x, 0)
        expx = np.exp(x - maxVal)
        sumExpx = np.sum(expx, 0) + 1e-7
        return expx / sumExpx
    def gradient(self, inputs, grad_value):
        x = inputs[0].value
        y = np.exp(x)
        sumy = np.sum(y, 0)
        return [(sumy - y.shape[0] * y) * y * grad_value / sumy / sumy]
    def __repr__(self):
        return "Softmax"
class matrix_crossEntropyLoss(Op):
    def __call__(self, a, label):
        return Node(self, [a, label])
    def compute(self, inputs):
        x, t = inputs[0].value, inputs[1].value
        return -np.sum(t * np.log(x + 1e-7)) / x.shape[1]
    def gradient(self, inputs, grad_value):
        x, t = inputs[0].value, inputs[1].value
        grad = np.zeros_like(x)
        grad[t == 1] = -1 / (x[t == 1] + 1e-7)
        return [grad_value * grad / x.shape[1], np.zeros_like(t)]
    def __repr__(self):
        return "CrossEntropyLoss"
class matrix_exp(Op):
    def __call__(self, a):
        return Node(self, [a])
    def compute(self, inputs):
        x = inputs[0].value
        x -= np.max(x, 0)
        return np.exp(x)
    def gradient(self, inputs, grad_value):
        x = inputs[0].value
        x -= np.max(x, 0)
        return [grad_value * np.exp(x)]
    def __repr__(self) -> str:
        return "matrix_exp"
class matrix_sqrt(Op):
    def __call__(self, a):
        return Node(self, [a])
    def compute(self, inputs):
        return np.sqrt(inputs[0].value)
    def gradient(self, inputs, grad_value):
        return [grad_value / np.sqrt(inputs[0].value) / 2]
    def __repr__(self) -> str:
        return "matrix_exp"
class matrix_columnSum(Op):
    def __call__(self, a):
        return Node(self, [a])
    def compute(self, inputs):
        return self._sumExt(inputs[0].value)
    def gradient(self, inputs, grad_value):
        return [self._sumExt(grad_value)]
    def __repr__(self) -> str:
        return "matrix_col_sum"
    def _sumExt(self, matrix):
        return np.sum(matrix, 0).reshape(1, matrix.shape[1]).repeat(matrix.shape[0], 0)
class matrix_rowSum(Op):
    def __call__(self, a):
        return Node(self, [a])
    def compute(self, inputs):
        return self._sumExt(inputs[0].value)
    def gradient(self, inputs, grad_value):
        return [self._sumExt(grad_value)]
    def __repr__(self) -> str:
        return "matrix_row_sum"
    def _sumExt(self, matrix):
        return np.sum(matrix, 1).reshape(matrix.shape[0], 1).repeat(matrix.shape[1], 1)
class matrix_div(Op):
    def __call__(self, a, b):
        return Node(self, [a, b])
    def compute(self, inputs):
        return inputs[0].value / inputs[1].value
    def gradient(self, inputs, grad_value):
        x, y = inputs[0].value, inputs[1].value
        return [grad_value / y, - grad_value * x / y / y]
    def __repr__(self):
        return "m/"
