import numpy as np
import math


class Linear():
    # DO NOT DELETE
    def __init__(self, in_feature, out_feature):
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.W = np.random.randn(out_feature, in_feature)
        self.b = np.zeros(out_feature)
        
        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x
        self.out = x.dot(self.W.T) + self.b
        return self.out

    def backward(self, delta):
        self.db = delta
        self.dW = np.dot(self.x.T, delta)
        dx = np.dot(delta, self.W.T)
        return dx

        

class Conv1D():
    def __init__(self, in_channel, out_channel, 
                 kernel_size, stride):

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        self.W = np.random.randn(out_channel, in_channel, kernel_size)
        self.b = np.zeros(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x
        batch_size = x.shape[0]
        self.in_width = x.shape[2]
        out = np.zeros((batch_size, self.out_channel, int((self.in_width - self.kernel_size) / self.stride + 1)))
        for channel in range(self.out_channel):
            for idx, width in enumerate(range(0, self.in_width - self.kernel_size + 1, self.stride)):
                out[:, channel, idx] = np.sum(
                    np.multiply(x[:, :, width: width + self.kernel_size], self.W[channel]) + self.b[channel],
                    axis=(1, 2))
        self.batch, __, self.width = x.shape
        assert __ == self.in_channel, 'Expected the inputs to have {} channels'.format(self.in_channel)
        #print(out.shape)
        return out


    def backward(self, delta):
        dx = np.zeros(self.x.shape)
        for inst in range(delta.shape[0]):
            for channel in range(self.out_channel):
                for idx, width in enumerate(range(0, self.in_width - self.kernel_size + 1, self.stride)):
                    dx[inst, :, width: width + self.kernel_size] += np.multiply(self.W[channel], delta[inst, channel, idx])
                    self.dW[channel] += (np.multiply(self.x[inst, :, width: width + self.kernel_size],
                                                     delta[inst, channel, idx]))
        self.db = np.sum(delta, axis=(0, 2))
        return dx




class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x_0 = x.shape[0]
        self.x_1 = x.shape[1]
        self.x_2 = x.shape[2]
        return x.reshape(x.shape[0], x.shape[1] * x.shape[2])

    def backward(self, x):
        return x.reshape(self.x_0, self.x_1, self.x_2)




class ReLU():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.dy = (x>=0).astype(x.dtype)
        return x * self.dy

    def backward(self, delta):
        return self.dy * delta