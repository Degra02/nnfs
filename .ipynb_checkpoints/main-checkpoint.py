from Layers.Dense import DenseLayer
from ActivationFunctions.ReLU import ReLu
from CreateData import spiral_data
import matplotlib.pyplot as plt

# Creating a random input (this is the input layer)
X, y = spiral_data(100, 3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
# plt.show()

layer1 = DenseLayer(2, 5)
activation1 = ReLu()

layer1.forward(X)
activation1.forward(layer1.output)
