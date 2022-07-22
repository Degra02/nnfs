import numpy as np

np.random.seed(0)


class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.output = None
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)  # so we don't need to transpose the weights matrix
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights + self.biases)
