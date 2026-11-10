import numpy as np
import matplotlib.pyplot as plt

class NeuralNet:
    def __init__(self, hidden=16, lr=0.5, epochs=2000, seed=0):

        rng = np.random.default_rng(seed)
        self.lr = lr
        self.epochs = epochs

        #initial weights
        self.W1 = rng.normal(0, 1, (2, hidden)) * 0.5 #mean = 0,  std = 0.5, shape = 2 x hidden
        self.b1 = np.zeros(hidden) 

        self.W2 = rng.normal(0, 1, (hidden, 1)) * 0.5 #mean = 0,  std =0.50, shape = hidden x 1
        self.b2 = np.zeros(1) 

        self.losses = []

    def _relu(self, z):
        return np.maximum(0,z)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _loss(self, Y, P):
        p = np.clip(P, 1e-12, 1 - 1e-12)
        return -np.mean(Y * np.log(p) + (1-Y)*np.log(1-p))

    def _forward(self, X):
        #multiply by W 
        Z1 = X @ self.W1 + self.b1 # X: nx2, W1: 2xhidden, b1: hidden, Z1 is now n x hidden
        A1 = self._relu(Z1) # relu all the entries of Z: n x hidden

        #multiply by W2
        Z2 = A1 @ self.W2 + self.b2 #A is n x hidden, W2 is hidden x 1, b2 is 1, Z2 is n x 1 
        #apply sigmoid
        P = self._sigmoid(Z2)

        #store z1 and A1 (needed for back prop)
        self.Z1 = Z1
        self.A1 = A1 

        return P 

    def _backwards(self, X, Y, P):
        n = X.shape[0]

        #gradients must match shape of paramters
        #average of the gradients for sigma p - y for each point
        dZ2 = (P - Y) / n    #(n , 1)

        #dL/dW2 = dL/dZ2 * dZ2/dW2
        dW2 = self.A1.T @ dZ2 # (H, n) @ (n, 1) -> (H, 1)
        #dL/dB2 = dL/Z2 * dZ2/db2 = dZ2 * 1 (single bias term summed and avg, already averaegd from dZ2)
        db2 = dZ2.sum(axis=0) # (1, )

        #dL/dA1 = dL/dZ2 * dZ2/dA1
        dA1 = dZ2 @ self.W2.T # (n, 1) @ (1, H) -> (n, H)
        #dL/dZ1 = dL/dA1 * dA1/dZ1. We use * since we arent mattrix multiplying here dA1/dZ1 is 1 when relu is active and 0 when its not
        dZ1 = dA1 * (self.Z1 > 0) # (n, H) * (n, H) -> (n, H) since no mattrxi multiply

        # dL/dW1 = dL/dZ1 * dZ1/dW1
        dW1 = X.T @ dZ1 #(2, n) @ (n, H) -> (2, H)
        #since dZ1/db1 = 1 
        db1 = dZ1.sum(axis=0) # (H, )

        return dW1, db1, dW2, db2

        #overall rule of thumg: linear layer -> transpose and use @
        #activation layer -> use * (since activations occur on each element directly so derrivative should as well)
        #avg happens at the top


    def fit(self, X, y):
        Y = y.reshape(-1,1) #go from (n, ) -> (n, 1) to match P
        self.losses = []
        for _ in range(self.epochs):
            P = self._forward(X)
            self.losses.append(self._loss(Y, P))
            dW1, db1, dW2, db2 = self._backwards(X, Y, P)

            #update weights
            self.W1 = self.W1 - self.lr * dW1
            self.b1 = self.b1 - self.lr * db1
            self.W2 = self.W2 - self.lr * dW2
            self.b2 = self.b2 - self.lr * db2

        return self


    def predict_proba(self, X):
        return self._sigmoid(self._relu (X @ self.W1 + self.b1) @ self.W2 + self.b2)

    def predict(self, X, thresh = 0.5):
            return (self.predict_proba(X) > thresh).astype(int).ravel() #ravel turns (n, 1) -> (n, )




        



def make_circles(n=200, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * np.pi, n)
    radii = np.where(np.arange(n) < n // 2, 1.0, 2.5)
    X = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    X += rng.normal(0, noise * 2, X.shape)
    y = (np.arange(n) >= n // 2).astype(int)
    return X, y

X, y = make_circles()
net = NeuralNet(hidden=16, lr=0.5, epochs=2000).fit(X, y)
print("first loss:", net.losses[0], " final loss:", net.losses[-1])
print("accuracy:", np.mean(net.predict(X) == y))

# decision boundary: predict on a grid, color by prediction
xx, yy = np.meshgrid(np.linspace(-4, 4, 200), np.linspace(-4, 4, 200))
grid = np.column_stack([xx.ravel(), yy.ravel()])
zz = net.predict_proba(grid).reshape(xx.shape)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(net.losses); ax[0].set_title("loss")
ax[1].contourf(xx, yy, zz, levels=[0, 0.5, 1], alpha=0.3, cmap="bwr")
ax[1].scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", s=10)
ax[1].set_title("decision boundary")
plt.show()