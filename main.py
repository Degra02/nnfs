from Layers.Dense import DenseLayer
from ActivationFunctions.ReLU import ReLu
from ActivationFunctions.SoftMax import SoftMax
from CreateData import spiral_data, linear_data
import matplotlib.pyplot as plt

# Creating a random input (this is the input layer)
X, y = spiral_data(samples=100, classes=3)
# X, y = linear_data(samples=100, classes=3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='jet')
plt.show()

dense1 = DenseLayer(2, 3)  # input is 2 because spiral_data generates [x, y] points
activation1 = ReLu()

dense2 = DenseLayer(3, 3)  # input is 3 because it's the output of activation1
activation2 = SoftMax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])
