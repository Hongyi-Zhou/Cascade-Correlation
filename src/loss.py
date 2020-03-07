import numpy as np
import os

# The following Criterion class will be used as the basis for a number
# of loss functions (which are in the form of classes so that they can be
# exchanged easily

class Criterion(object):
    """
    Interface for loss functions.
    """
    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented

class SoftmaxCrossEntropy(Criterion):
    """
    Softmax loss
    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()
        self.softmax = None

    def forward(self, x, y):
        """
        Argument:
            x (np.array): (batch size, 10)
            y (np.array): (batch size, 10)
        Return:
            out (np.array): (batch size, )
        """
        self.logits = x
        self.labels = y
        num = len(x)
        k = y.argmax(axis=1)
        exps = np.exp(x)
        div = np.sum(exps, axis = 1)
        self.softmax = exps/div[:, np.newaxis]
        self.loss = -np.log(self.softmax[range(num),k])

        return self.loss

    def derivative(self):
        """
        Return:
            out (np.array): (batch size, 10)
        """
        num = len(self.softmax)
        grad = self.softmax
        k = self.labels.argmax(axis=1)
        grad[range(num), k] -= 1

        return grad  

class L1Loss(Criterion):
    """
    L1 Loss: Mean Absolute Error
    """

    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, x, y):
        """
        Argument:
            x (np.array): (batch size, 10)
            y (np.array): (batch size, 10)
        Return:
            out (np.array): (batch size, )
        """
        return abs(x-y)

    def backward(self):
        pass

class L2Loss(Criterion):
    """
    L2 Loss: Mean Square Error
    """

    def __init__(self):
        super(L2Loss, self).__init__()
        self.x = None
        self.y = None

    def forward(self, x, y):
        """
        Argument:
            x (np.array): (batch size, 10)
            y (np.array): (batch size, 10)
        Return:
            out (np.array): (batch size, )
        """
        self.x = x
        self.y = y
        return np.sum((x-y)**2)/2

    def backward(self):
        return (self.x - self.y)

class KL_Divergence(Criterion):
    """
    Kullback-Leibler divergence
    """
    def __init__(self):
        super(KL_Divergence, self).__init__()

    def forward(self, x, y):
        """
        Argument:
            x (np.array): (batch size, 10)
            y (np.array): (batch size, 10)
        Return:
            out (np.array): (batch size, )
        """
        return y*np.log(y/x)

    def backward(self):
        pass


