import numpy as np


class Conv1D():
    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)
        
        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

        self.input = None;

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
        """
        
        self.input = x

        batch_size, in_channel, input_size = x.shape
        self.input_size = input_size
        output_size = int((input_size - self.kernel_size)/self.stride) + 1
        
        out = np.zeros((batch_size, self.out_channel, output_size))
        for i in range(batch_size):
            for j in range(self.out_channel):
                for k in range(output_size):
                    out[i,j,k] += np.sum(x[i,:,k*self.stride:k*self.stride+self.kernel_size] * self.W[j,:,:]) + self.b[j]
        return out



    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_size)
        Return:
            dx (np.array): (batch_size, in_channel, input_size)
        """
        batch_size, out_channel, output_size = delta.shape
        batch_size, in_channel, input_size = self.input.shape
        dx = np.zeros((batch_size, in_channel, input_size))

        for i in range(batch_size):
            for j in range(out_channel):
                curr_ = out_ = 0
                while curr_ + self.kernel_size <= input_size:
                    self.dW[j] += delta[i,j,out_] * self.input[i,:, curr_:curr_ + self.kernel_size]
                    dx[i, :, curr_:curr_ + self.kernel_size] += delta[i,j,out_] * self.W[j]
                    curr_ += self.stride
                    out_ += 1

                self.db[j] += np.sum(delta[i,j,:])

        return dx



class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """
        self.b, self.c, self.w = x.shape
        return x.reshape((self.b, self.c*self.w))

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in channel * in width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """
        return delta.reshape((self.b, self.c, self.w))
