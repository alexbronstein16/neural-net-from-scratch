import numpy as np

class NeuralNet:
   
    def __init__(self, layers=(2,16,1), lr=0.5, epochs=2000, batch_size=None, seed=0):
      
        rng = np.random.default_rng(seed)
        self.layers = layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

        #check if we are doing multiple categories or just 1:
        if layers[-1] > 1:
            self.cat = True
        else:
            self.cat = False

        #layers defines architectire: layers[0] is the input size / how many features 
        self.Ws, self.Bs = [], []
        self.losses = []
        for i in range(len(layers) - 1):
            #initialize weights He init with std = sqrt(2 / fan_in) 
            W = rng.normal(0, 1, (layers[i], layers[i+1])) * np.sqrt(2 / layers[i]) #layers[i] inputs -> layers[i + 1] neurons each neuron needs layers[i] weights + bias
            b = np.zeros(layers[i+1]) #i+1 biases 

            self.Ws.append(W)
            self.Bs.append(b)

    def _relu(self, Z):
        return np.maximum(0,Z)

    def _sigmoid(self, Z):
        return (1 / (1 + np.exp(-Z)))

    def _softmax(self, Z):
        Z = Z - Z.max(axis=1, keepdims=True) #for each point find the "category with highest score" then subtract that from every point in row
        #every row now has 0 as largest so e^z wont be huge
        expz = np.exp(Z)
        tot = expz.sum(axis=1, keepdims=True) #get the total for each row
        return expz / tot


    def _loss(self, Y, p):
        P = np.clip(p, 1e-12, 1 - 1e-12)
        #different loss for multiple categories or just binary
        if self.cat:
            return -np.mean(np.sum(Y * np.log(P), axis=1)) #p has n points with k classes so we need to sum across k and not n
        else:
            return -np.mean(Y * np.log(P) + (1-Y)* np.log(1-P))


    def _forward(self, X):
        Curr = X

        self.Zs = []
        self.As = [X]
        for i in range(len(self.Ws) - 1):
            Zi = Curr @ self.Ws[i] + self.Bs[i]
            Ai = self._relu(Zi)
            Curr = Ai

            #store Zi and Ai
            self.Zs.append(Zi)
            self.As.append(Ai)

        Zlast = Curr @ self.Ws[-1] + self.Bs[-1]

        if self.cat:
            P = self._softmax(Zlast)
        else:
            P = self._sigmoid(Zlast)
        return P

    def _backwards(self, X, Y, P):
        n = X.shape[0]
        dWs = [None] * len(self.Ws)
        dBs = [None] * len(self.Bs)
        dZ = (P - Y) / n #(n , k)
        
        for i in reversed(range(len(self.Ws))):
            dWs[i] = self.As[i].T @ dZ
            dBs[i] = dZ.sum(axis=0) 
            #pass error unless on last pass, we dont need these since we don't have antyhing to pass back
            if i > 0:
                dA = dZ @ self.Ws[i].T 
                dZ = dA * (self.Zs[i-1] > 0) #element wise multiplication by relu' 0 if negative and slope is 1 if positive

        return dWs, dBs

    def _encode(self, y):
        if self.cat:
            K = self.layers[-1] 
            #Y labels are non binary here so we want to turn to binary 2 -> 0 0 1 0 0 ...
            Y = np.eye(K)[y] #go from (n, ) -> (n, K) to match P
        else:
            Y = y.reshape(-1, 1) #go from (n, ) -> (n, 1) to match P
        return Y

    def _labels(self, P, thresh = 0.5):
        if self.cat:
            return P.argmax(axis=1) #(n,) index of the highest probability per row
        else:
            return (P > thresh).astype(int).ravel() #ravel turns (n, 1) -> (n, )


    def fit(self, X, y,  X_val=None, y_val=None, log_every=1):
        Y = self._encode(y)
        n = X.shape[0]
        
        bs = self.batch_size or n #either batch size or full data set 
        rng = np.random.default_rng(self.seed)
        self.losses = []
        self.history = {"epoch": [], "train_acc": [], "val_acc": [], "val_loss": []}
    
        #each epoch is 1 pass over our shuffled batches
        for epoch in range(self.epochs):
            idx = rng.permutation(n) #shuffled sequcnes from 0 to n-1
            for start in range(0, n, bs): #start stop step
                b = idx[start:start + bs] #getting indexes of the ones in this batch 
                Xb, Yb = X[b], Y[b] #getting our data and labels for this batch 
                P = self._forward(Xb)
                dWs, dBs = self._backwards(Xb, Yb, P)
            
                for i in range(len(self.Ws)):
                    #update weights and biases 
                    self.Ws[i] = self.Ws[i] - self.lr * dWs[i]
                    self.Bs[i] = self.Bs[i] - self.lr * dBs[i]
            #end of epoch
            P_all = self._forward(X)
            self.losses.append(self._loss(Y, P_all))

            #logging assuming we have validation and we are on intervals of log_every or on last epoch
            if X_val is not None and (epoch % log_every == 0 or epoch == self.epochs - 1): 
                P_val = self.predict_proba(X_val)
                self.history["epoch"].append(epoch)
                self.history["train_acc"].append(np.mean(self._labels(P_all) == y))
                self.history["val_acc"].append(np.mean(self._labels(P_val) == y_val))
                self.history["val_loss"].append(self._loss(self._encode(y_val), P_val))
                print(f"epoch {epoch:3d}  loss {self.losses[-1]:.4f}  "
                    f"train acc {self.history['train_acc'][-1]:.4f}  "
                    f"val acc {self.history['val_acc'][-1]:.4f}")
                
        return self

    def predict_proba(self, X):
       return self._forward(X)

    def predict(self, X, thresh = 0.5):
        P = self.predict_proba(X)
        return self._labels(P, thresh)