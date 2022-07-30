from CreateData import vertical_data
from Layers.Dense import DenseLayer
from ActivationFunctions.ReLU import ReLu
from ActivationFunctions.SoftMax import SoftMax
from Loss import Loss_CategoricalCrossEntropy
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange

X, y = vertical_data(samples=100, classes=3)

plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
plt.show()

# Trying to reduce Loss by using random weights and biases and
# using the lowest Loss obtained for the vertical dataset

dense1 = DenseLayer(2, 3)
activation1 = ReLu()
dense2 = DenseLayer(3, 3)
activation2 = SoftMax()

loss_function = Loss_CategoricalCrossEntropy()
lowest_loss = 999999

best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

progress = trange(100000)  # 100k iteration to "find" the lowest loss
progress.set_description(f"")
for i in progress:
    dense1.weights = 0.05 * np.random.randn(2, 3)  # passing the required shape to the randn method
    dense1.biases = 0.05 * np.random.randn(1, 3)
    dense2.weights = 0.05 * np.random.randn(3, 3)
    dense2.biases = 0.05 * np.random.randn(1, 3)

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss = loss_function.calculate(activation2.output, y)

    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)
    
    progress.set_description(f"Iteration: {i+1}, Loss: {loss}, Lowest Loss: {lowest_loss}, Accuracy: {accuracy}")
    if loss < lowest_loss:
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss