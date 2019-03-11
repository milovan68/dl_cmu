"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)
import numpy as np
import os


class Activation(object):

    """
    Interface for activation functions (non-linearities).

    In all implementations, the state attribute must contain the result, i.e. the output of forward (it will be tested).
    """

    # No additional work is needed for this class, as it acts like an abstract base class for the others

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):

    """
    Identity function (already implemented).
    """

    # This class is a gimme as it is already implemented for you as an example

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):

    """
    Sigmoid non-linearity
    """

    # Remember do not change the function signatures as those are needed to stay the same for AL

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        self.state = 1 / (1 + np.exp(-x))

        # Might we need to store something before returning?

        return self.state

    def derivative(self):

        # Maybe something we need later in here...

        return self.state * (1 - self.state)


class Tanh(Activation):

    """
    Tanh non-linearity
    """

    # This one's all you!

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        self.state = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return self.state

    def derivative(self):
        return 1  - self.state * self.state


class ReLU(Activation):

    """
    ReLU non-linearity
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        x[x <= 0] = 0
        self.state = x
        return self.state

    def derivative(self):
        relu_derivative = self.state.copy()
        relu_derivative[relu_derivative <= 0] = 0
        relu_derivative[relu_derivative > 0] = 1
        return relu_derivative

# Ok now things get decidedly more interesting. The following Criterion class
# will be used again as the basis for a number of loss functions (which are in the
# form of classes so that they can be exchanged easily (it's how PyTorch and other
# ML libraries do it))


class Criterion(object):

    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

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
        self.sm = None

    def forward(self, x, y):

        self.logits = x
        self.labels = y
        shift_logits = self.logits - np.max(self.logits, axis=1)[:, np.newaxis]
        self.sm = np.exp(shift_logits) / np.sum(np.exp(shift_logits), axis=1)[:, np.newaxis]
        return -1 * np.sum(self.labels * np.log(self.sm), axis=1)

    def derivative(self):

        # self.sm might be useful here...

        return self.sm - self.labels


class BatchNorm(object):

    def __init__(self, fan_in, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, fan_in))
        self.mean = np.zeros((1, fan_in))

        self.gamma = np.ones((1, fan_in))
        self.dgamma = np.zeros((1, fan_in))

        self.beta = np.zeros((1, fan_in))
        self.dbeta = np.zeros((1, fan_in))

        # inference parameters
        self.running_mean = np.zeros((1, fan_in))
        self.running_var = np.ones((1, fan_in))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):

        if eval:
            self.norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            return self.gamma * self.norm + self.beta

        self.x = x
        self.mean = np.mean(self.x, axis=0)
        self.var = 1. / self.x.shape[0] * np.sum((self.x - self.mean) ** 2, axis=0)
        self.norm = (self.x - self.mean) / np.sqrt(self.var + self.eps)
        self.out = self.gamma * self.norm + self.beta

        # update running batch statistics
        self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * self.mean
        self.running_var = self.alpha * self.running_var + (1 - self.alpha) * self.var
        return self.out

        # ...


    def backward(self, delta):
        self.dbeta = np.sum(delta, axis=0) * delta.shape[0]
        self.dgamma = np.sum(((self.x - self.mean) * (self.var + self.eps) ** (-1. / 2.)) * delta, axis=0) * delta.shape[0]
        dx = (1 / delta.shape[0]) * self.gamma * (self.var + self.eps) ** (-1. / 2.) * (delta.shape[0] * delta - np.sum(delta, axis=0) - (self.x - self.mean) * (self.var + self.eps) ** (-1.0) * np.sum(delta * (self.x - self.mean), axis=0))
        return dx


# These are both easy one-liners, don't over-think them
def random_normal_weight_init(d0, d1):
    return np.random.randn(d0, d1)


def zeros_bias_init(d):
    return np.zeros((1, d))


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn, bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.activations_state = []
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        self.zeros = bias_init_fn
        self.batch_size = None

        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly
        self.neurons = [input_size] + hiddens + [output_size]
        self.W = [weight_init_fn(x, y) for x, y in zip(self.neurons[:-1], self.neurons[1:])]
        self.dW = [np.zeros((x, y)) for x, y in zip(self.neurons[:-1], self.neurons[1:])]
        self.b = [self.zeros(x) for x in self.neurons[1:]]
        self.db = [self.zeros(x) for x in self.neurons[1:]]
        self.momentum_W = [np.zeros((x, y)) for x, y in zip(self.neurons[:-1], self.neurons[1:])]
        self.momentum_b = [self.zeros(x) for x in self.neurons[1:]]
        # HINT: self.foo = [ bar(???) for ?? in ? ]

        # if batch norm, add batch norm parameters
        if self.bn:
            self.bn_layers = [BatchNorm(self.neurons[i + 1]) for i in range(self.num_bn_layers)]

        # Feel free to add any other attributes useful to your implementation (input, output, ...)

    def forward(self, x):
        self.batch_size = x.shape[0]
        self.forward_activation = x
        self.activations_state.append(self.forward_activation)
        for idx, W in enumerate(self.W):
            self.forward_activation = np.dot(self.forward_activation, W) + self.b[idx]
            if self.bn and idx == 0:
                self.forward_activation = self.bn_layers[idx].forward(self.forward_activation,
                                                                      eval=not(self.train_mode))
            activ_function = self.activations[idx]
            self.forward_activation = activ_function.forward(self.forward_activation)
            self.activations_state.append(self.forward_activation)
        return self.forward_activation



    def zero_grads(self):
        self.dW = [np.zeros((x, y)) for x, y in zip(self.neurons[:-1], self.neurons[1:])]
        self.db = [self.zeros(x) for x in self.neurons[1:]]

    def step(self):
        self.momentum_W = [moment_W * self.momentum - self.lr * dW for dW, moment_W in zip(self.dW, self.momentum_W)]
        self.momentum_b = [moment_b * self.momentum - self.lr * db for db, moment_b in zip(self.db, self.momentum_b)]
        self.W = [W + momentum for W, momentum in zip(self.W, self.momentum_W)]
        self.b = [b + momentum for b, momentum in zip(self.b, self.momentum_b)]
        if self.bn:
            self.bn_layers[0].gamma -= self.lr * self.bn_layers[0].dgamma
            self.bn_layers[0].beta -= self.lr * self.bn_layers[0].dbeta


    def backward(self, labels):
        loss = self.criterion.forward(self.forward_activation, labels)
        loss_grad = (self.criterion.derivative() * self.activations[-1].derivative()) / self.batch_size
        self.db[-1] = np.sum(loss_grad, axis=0)
        self.dW[-1] = np.dot(self.activations_state[-2].T, loss_grad)
        for i in range(2, len(self.neurons)):
            loss_grad = (np.dot(loss_grad, self.W[-i + 1].T) * self.activations[-i].derivative())
            if self.bn and len(self.bn_layers) == len(self.neurons) - i:
                loss_grad = self.bn_layers[0].backward(loss_grad)
            self.db[-i] = np.sum(loss_grad, axis=0)
            self.dW[-i] = np.dot(self.activations_state[-i - 1].T, loss_grad)
        self.activations_state = []


    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False


def weight_init(x, y):
    return np.random.randn(x, y)

def bias_init(x):
    return np.zeros((1, x))


def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, test = dset
    trainx, trainy = train
    valx, valy = val
    testx, testy = test

    idxs = np.arange(len(trainx))

    training_losses = []
    training_errors = []
    validation_losses = []
    validation_errors = []


    for e in range(nepochs):
        print(e)
        mlp.zero_grads()
        s = np.arange(trainx.shape[0])
        np.random.shuffle(s)
        trainx, trainy = trainx[s], trainy[s]

        # Per epoch setup ...
        error_train = 0
        loss_train = 0
        error_val = 0
        loss_val = 0
        count_train = 0
        count_val = 0
        for b in range(0, len(trainx), batch_size):
            mlp.train()
            count_train += 1
            mlp.zero_grads()
            pred = mlp.forward(trainx[b:b + batch_size])
            cross = SoftmaxCrossEntropy()
            loss_train += np.mean(cross.forward(pred, trainy[b:b + batch_size]))
            error_train += 1 - np.sum(np.argmax(pred, axis=1) == np.argmax(trainy[b:b + batch_size], axis=1)) / batch_size
            mlp.backward(trainy[b:b + batch_size])
            mlp.step()


        for b in range(0, len(valx), batch_size):
            count_val += 1
            mlp.eval()
            pred = mlp.forward(valx[b:b + batch_size])
            cross = SoftmaxCrossEntropy()
            loss_val += np.mean(cross.forward(pred, valy[b:b + batch_size]))
            error_val += 1 - np.sum(np.argmax(pred, axis=1) == np.argmax(valy[b:b + batch_size], axis=1)) / batch_size
            # Remove this line when you start implementing this
            # Val ...
        training_losses.append(loss_train / count_train)
        training_errors.append(error_train / count_train)
        validation_losses.append(loss_val / count_val)
        validation_errors.append(error_val / count_val)
        # Accumulate data...

    # Cleanup ...

    for b in range(0, len(testx), batch_size):

        mlp.eval()
        pred = mlp.forward(testx[b:b + batch_size])
        pred_digits = np.argmax(pred, axis=1)

    # Return results ...

    return (training_losses, training_errors, validation_losses, validation_errors)

