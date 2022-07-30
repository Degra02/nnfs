import numpy as np
np.random.seed(0)


def sig(x):
    return 1 / (1 + np.exp(-x))


# generates [x, y] points, so output X.dim=2, while y is the number of classes
def spiral_data(samples, classes):
    X = np.zeros((samples * classes, 2))
    y = np.zeros(samples * classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples * class_number, samples * (class_number + 1))
        r = np.linspace(0.0, 1, samples)  # radius
        t = np.linspace(class_number * 4, (class_number + 1) * 4, samples) + np.random.randn(samples) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y


def linear_data(samples, classes, start=0.0, end=1):
    X = np.zeros((samples * classes, 2))
    y = np.zeros(samples * classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples * class_number, samples * (class_number + 1))
        r = np.linspace(start, end, samples)
        t = np.random.randn(samples)
        z = np.random.randn(samples)
        X[ix] = np.c_[r + sig(z), r + sig(t)]
        y[ix] = class_number
    return X, y


def vertical_data(samples, classes):
    X = np.zeros((samples*classes, 2))
    y = np.zeros(samples*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples*class_number, samples*(class_number+1))
        X[ix] = np.c_[np.random.randn(samples)*.1 +     class_number/3, np.random.randn(samples)*.1 + 0.5]
        y[ix] = class_number
    return X, y
