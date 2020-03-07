import numpy as np
import math

class Linear():
    def __init__(self, in_feature, out_feature, weight_init_fn, bias_init_fn):

        """
        Argument:
            W (np.array): (in feature, out feature)
            dW (np.array): (in feature, out feature)
            momentum_W (np.array): (in feature, out feature)

            b (np.array): (1, out feature)
            db (np.array): (1, out feature)
            momentum_B (np.array): (1, out feature)
        """

        self.W = weight_init_fn(in_feature, out_feature)
        self.b = bias_init_fn(out_feature)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

        self.momentum_W = np.zeros(None)
        self.momentum_b = np.zeros(None)
        
        self.output = None
        self.input = None
        

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, out feature)
        """
        self.input = x
        self.output = np.dot(x, self.W) + self.b
        return self.output
    
    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, out feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        
        self.db = (np.sum(delta, axis = 0)/delta.shape[0]).reshape(1,-1)
        self.dW = np.dot(self.input.T, delta)/delta.shape[0]
        
        return np.dot(delta, self.W.T)
