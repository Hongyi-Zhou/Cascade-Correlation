import numpy as np

class BatchNorm(object):

    def __init__(self, in_feature, alpha=0.9):

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        self.var = np.ones((1, in_feature))
        self.mean = np.zeros((1, in_feature))

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))

        # inference parameters
        self.running_mean = np.zeros((1, in_feature))
        self.running_var = np.ones((1, in_feature))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        """
        Argument:
            x (np.array): (batch_size, in_feature)
            eval (bool): inference status

        Return:
            out (np.array): (batch_size, in_feature)
        """

        self.x = x
        if eval:
            self.mean = self.running_mean
            self.var = self.running_var
            self.norm = (self.x - self.mean)/np.sqrt(self.var + self.eps)
            self.out = self.gamma * self.norm + self.beta
            return self.out
        else:
    
            self.mean = np.mean(x, axis = 0, keepdims = True)
            self.var = np.var(x, axis = 0, keepdims = True)
            self.norm = (self.x - self.mean)/np.sqrt(self.var + self.eps)
            self.out = self.gamma * self.norm + self.beta
    
            # Update running batch statistics
            self.running_mean = self.alpha * self.running_mean + (1-self.alpha) * self.mean
            self.running_var = self.alpha * self.running_var + (1-self.alpha) * self.var
    
            return self.out


    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        dx_norm = delta * self. gamma
        dvar = -0.5 * np.sum(dx_norm * (self.x - self.mean), axis = 0) * ((self.var + self.eps)**(-3.0/2))
        dmean = - np.sum(dx_norm, axis = 0) /np.sqrt(self.var + self.eps) - (2.0/len(self.x)) * dvar * np.sum(self.x - self.mean,axis = 0)
        dx = (dx_norm/np.sqrt(self.var + self.eps)) + (dvar*2/len(self.x) *(self.x-self.mean)) + (dmean/len(self.x))
        
        self.dbeta = np.sum(delta, axis = 0, keepdims = True)
        self.dgamma = np.sum(delta * self.norm, axis = 0, keepdims = True)
        
        return dx
