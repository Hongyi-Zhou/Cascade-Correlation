import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from batchnorm import *
from linear import *


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn,
                 bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum

        self.output = None

        layers = [input_size] + hiddens + [output_size]
        self.linear_layers = [Linear(layers[i], layers[i+1], weight_init_fn, bias_init_fn) for i in range(len(layers)-1)]

        if self.bn:
            self.bn_layers = []
            for i in range(self.num_bn_layers):
                self.bn_layers.append(BatchNorm(layers[i+1]))


    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, input_size)
        Return:
            out (np.array): (batch size, output_size)
        """

        ret_prev = x
        for i in range(self.nlayers):
            ret_linear = self.linear_layers[i].forward(ret_prev)
            if self.bn and len(self.bn_layers) > i:
                ret_bn = self.bn_layers[i].forward(ret_linear, eval = not self.train_mode)
            else:
                ret_bn = ret_linear
                
            ret_act = self.activations[i].forward(ret_bn)
            ret_prev = ret_act
        self.output = ret_act
        
        return self.output
        

    def zero_grads(self):
        # Use numpyArray.fill(0.0) to zero out your backpropped derivatives in each
        # of your linear and batchnorm layers.
        for i in range(len(self.linear_layers)):
            # Update weights and biases here
            self.linear_layers[i].dW.fill(0.0)
            self.linear_layers[i].db.fill(0.0)

        if (self.bn_layers):
            for i in range(len(self.bn_layers)):
                self.bn_layers[i].dgamma.fill(0.0)
                self.bn_layers[i].dbeta.fill(0.0)

    def step(self):

        for i in range(len(self.linear_layers)):
            # Update weights and biases here
            #linear layer
            self.linear_layers[i].momentum_W = self.momentum * self.linear_layers[i].momentum_W - self.lr * self.linear_layers[i].dW
            self.linear_layers[i].W += self.linear_layers[i].momentum_W
            self.linear_layers[i].momentum_b = self.momentum * self.linear_layers[i].momentum_b - self.lr * self.linear_layers[i].db
            self.linear_layers[i].b += self.linear_layers[i].momentum_b
     
            #batchnorm layer
            if self.bn and len(self.bn_layers) > i:
                self.bn_layers[i].beta -= self.lr * self.bn_layers[i].dbeta
                self.bn_layers[i].gamma -= self.lr * self.bn_layers[i].dgamma


    def backward(self, labels):
        # Backpropagate through the activation functions, batch norm and
        # linear layers.
        self.zero_grads()
        self.criterion(self.output, labels)
        gradient = self.criterion.derivative()        
        
        prev_grad = gradient
        for i in range(self.nlayers-1, -1, -1):
            act_grad = self.activations[i].derivative() * prev_grad
            if self.bn and len(self.bn_layers) > i:
                bn_grad = self.bn_layers[i].backward(act_grad)  
            else:
                bn_grad = act_grad
            linear_grad = self.linear_layers[i].backward(bn_grad)            
            prev_grad = linear_grad

    def error(self, labels):
        return (np.argmax(self.output, axis = 1) != np.argmax(labels, axis = 1)).sum()

    def total_loss(self, labels):
        return self.criterion(self.output, labels).sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, _ = dset
    trainx, trainy = train
    valx, valy = val

    idxs = np.arange(len(trainx))

    training_losses = np.zeros(nepochs)
    training_errors = np.zeros(nepochs)
    validation_losses = np.zeros(nepochs)
    validation_errors = np.zeros(nepochs)

    # Setup ...

    for e in range(nepochs):
        np.random.shuffle(idxs)
        # Per epoch setup ...
        train_total_loss = 0
        train_total_error = 0
        val_total_loss = 0
        val_total_error = 0
        
        for b in range(0, len(trainx), batch_size):
            # Train ...
            mlp.train()
            data = trainx[idxs[b:b+batch_size]] if b+batch_size <= len(trainx) else trainx[idxs[b:]]
            label = trainy[idxs[b:b+batch_size]] if b+batch_size <= len(trainx) else trainy[idxs[b:]]
            
            mlp.zero_grads()
            mlp(data)
            mlp.backward(label)
            mlp.step()
            train_total_loss += mlp.total_loss(label)
            train_total_error += mlp.error(label)
            

        for c in range(0, len(valx), batch_size):
            # Val ...
            mlp.eval()
            val_data = valx[c:c+batch_size] if c+batch_size <= len(valy) else valx[c:]
            val_label = valy[c:c+batch_size] if c+batch_size <= len(valy) else valy[c:]
            
            mlp(val_data)
            val_total_loss += mlp.total_loss(val_label)
            val_total_error += mlp.error(val_label)

        # Accumulate data...
        training_losses[e] = train_total_loss/(b+1)
        training_errors[e] = train_total_error/len(trainx)
        validation_losses[e] = val_total_loss/(c+1)
        validation_errors[e] = val_total_error/len(valx)
        print("epochs:", e, "  tl:", training_losses[e])
    # Return results ...

    return (training_losses, training_errors, validation_losses, validation_errors)

