import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from neuralnet import NeuralNet


digits = load_digits()
X, y = digits.data / 16.0, digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y)

net = NeuralNet(layers=(64, 32, 10), lr=0.1, epochs=50, batch_size=32)
net.fit(X_train, y_train, X_test, y_test, log_every=10)
print("test acc:", np.mean(net.predict(X_test) == y_test))

