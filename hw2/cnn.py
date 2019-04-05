from layers import *

class CNN_B():
    def __init__(self):
        # Your initialization code goes here
        self.layers = []
        self.layers.append(Conv1D(24, 8, 8, 4))
        self.layers.append(ReLU())
        self.layers.append(Conv1D(8, 16, 1, 1))
        self.layers.append(ReLU())
        self.layers.append(Conv1D(16, 4, 1, 1))
        self.layers.append(Flatten())

    def __call__(self, x):
        return self.forward(x)

    def init_weights(self, weights):
        self.layers[0].W = weights[0].reshape(8, 24, 8).T
        self.layers[2].W = weights[1].T.reshape(16, 8, 1)
        self.layers[4].W = weights[2].T.reshape(4, 16, 1)
        # Load the weights for your CNN from the MLP Weights given

    def forward(self, x):
        # You do not need to modify this method
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        # You do not need to modify this method
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta




class CNN_C():
    def __init__(self):
        # Your initialization code goes here
        self.layers = []
        self.layers.append(Conv1D(24, 8, 8, 4))
        self.layers.append(ReLU())
        self.layers.append(Conv1D(8, 16, 1, 1))
        self.layers.append(ReLU())
        self.layers.append(Conv1D(16, 4, 1, 1))
        self.layers.append(Flatten())

    def __call__(self, x):
        return self.forward(x)

    def init_weights(self, weights):
        for i in range(weights.shape[0]):
            weights[i][np.isnan(weights[i])] = 0
        self.layers[0].W = weights[0].reshape(8, 24, 8).T
        self.layers[2].W = weights[1].T.reshape(16, 8, 1)
        self.layers[4].W = weights[2].T.reshape(4, 16, 1)
        # Load the weights for your CNN from the MLP Weights given


    def forward(self, x):
        # You do not need to modify this method
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        # You do not need to modify this method
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta
