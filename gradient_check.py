"""Check backprop against finite differences.

For a few weights in each layer, nudge the weight up and down by a tiny eps,
recompute the loss, and compare (loss_plus - loss_minus) / (2 * eps) with the
gradient from _backwards. If backprop is correct, the two numbers match.
"""
import numpy as np
from neuralnet import NeuralNet


def check(net, X, y, n_weights=3, eps=1e-5, seed=0):
    rng = np.random.default_rng(seed)
    Y = net._encode(y)
    P = net._forward(X)
    dWs, _ = net._backwards(X, Y, P)

    worst = 0.0
    for layer, W in enumerate(net.Ws):
        for _ in range(n_weights):
            r, c = rng.integers(W.shape[0]), rng.integers(W.shape[1])
            orig = W[r, c]
            W[r, c] = orig + eps
            loss_plus = net._loss(Y, net._forward(X))
            W[r, c] = orig - eps
            loss_minus = net._loss(Y, net._forward(X))
            W[r, c] = orig

            numerical = (loss_plus - loss_minus) / (2 * eps)
            analytic = dWs[layer][r, c]
            worst = max(worst, abs(numerical - analytic))
            print(f"layer {layer} W[{r},{c}]  numerical {numerical: .8f}  backprop {analytic: .8f}")
    print(f"largest difference: {worst:.2e}\n")


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("binary output (sigmoid), layers (2, 16, 8, 1)")
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int)
    check(NeuralNet(layers=(2, 16, 8, 1)), X, y)

    print("multi-class output (softmax), layers (64, 32, 10)")
    X = rng.random((300, 64))
    y = rng.integers(0, 10, 300)
    check(NeuralNet(layers=(64, 32, 10)), X, y)
