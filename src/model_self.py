import os,sys,time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class MLP:
    def __init__(self, input_size, hidden_layers, output_size=10, lr=0.001):
        self.input_size = input_size
        if isinstance(hidden_layers, int):
            self.hidden_layers = [hidden_layers]
        else: self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.lr = lr

        # weight and bais
        self.W=[]
        self.B=[]
        self.W.append(np.random.randn(input_size, hidden_layers[0]) * 0.01)
        self.B.append(np.zeros((1, hidden_layers[0])))
        for i in range(1, len(hidden_layers)):
            self.W.append(np.random.randn(hidden_layers[i-1], hidden_layers[i]) * 0.01)
            self.B.append(np.zeros((1, hidden_layers[i])))
        
        self.W.append(np.random.randn(hidden_layers[-1], output_size) * 0.01)
        self.B.append(np.zeros((1, output_size)))

        # Momentum
        self.mW=[np.zeros_like(w) for w in self.W]
        self.mB=[np.zeros_like(b) for b in self.B]
        self.beta1=0.9

        # Adam
        self.vW=[np.zeros_like(w) for w in self.W]
        self.vB=[np.zeros_like(b) for b in self.B]
        self.beta2=0.999
        self.eps=1e-8
        self.t=0


    def ReLU(self, z):
        return np.maximum(0, z)
    def ReLU_grad(self, z):
        return (z > 0).astype(float)
    def Softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def Loss(self, y_true, y_pred):
        # assert y_true.shape==y_pred.shape
        probabilty=np.sum(y_true*y_pred,axis=1)
        return -np.mean(np.log(probabilty+1e-9))


        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), np.argmax(y_true, axis=1)])
        loss = np.sum(log_likelihood) / m
        return loss

    def forward(self,x):
        self.Z=[]
        self.A=[]
        temp=x
        for i in range(len(self.W)-1):
            z= temp @ self.W[i] + self.B[i]
            temp=self.ReLU(z)
            self.Z.append(z)
            self.A.append(temp)

        z= temp @ self.W[-1] + self.B[-1]
        self.Z.append(z)

        return self.Softmax(z)
    
    def backward(self, x, y, output):
        m = y.shape[0]
        delta = output - y
        for i in reversed(range(len(self.W))):
            W_current = self.W[i].copy()

            if i==0:
                a_prev = x
            else:
                a_prev = self.A[i-1]

            dW = (a_prev.T @ delta)/m
            db = np.sum(delta, axis=0, keepdims=True) / m
            

            self.t+=1

            # moment
            self.mW[i] = self.beta1 * self.mW[i] + (1 - self.beta1) * dW
            self.mB[i] = self.beta1 * self.mB[i] + (1 - self.beta1) * db

            # ADAM
            self.vW[i] = self.beta2 * self.vW[i] + (1 - self.beta2) * (dW**2)
            self.vB[i] = self.beta2 * self.vB[i] + (1 - self.beta2) * (db**2)

            mW_hat=self.mW[i]/(1-self.beta1**self.t)
            mB_hat=self.mB[i]/(1-self.beta1**self.t)

            vW_hat=self.vW[i]/(1-self.beta2**self.t)
            vB_hat=self.vB[i]/(1-self.beta2**self.t)
            
            self.W[i] -= self.lr * mW_hat /(np.sqrt(vW_hat)+self.eps)
            self.B[i] -= self.lr * mB_hat /(np.sqrt(vB_hat)+self.eps)

            # self.W[i] -= self.lr * dW
            # self.B[i] -= self.lr * db

            if i > 0:
                delta = (delta @ W_current.T) * self.ReLU_grad(self.Z[i-1])
    
    def __repr__(self):
        text=f"{__class__.__name__}: [\n"
        text+=f" Input-size: {self.input_size},\n Output-size: {self.output_size},\n num hidden layer: {len(self.hidden_layers)},\n LR: {self.lr}"
        text+=f"\n ]\n\n"

        text+=f"Short hand transformation: [\n"
        text+=f" {self.input_size} -> "
        for i in self.hidden_layers:
            text+=f"{i} -> "
        text+=f"{self.output_size}"
        text+=f"\n ]\n\n"
        return text

if __name__=="__main__":
    x=np.random.randint(0,256,size=(1, 28,28),dtype=np.uint8)/255.0
    
    model=MLP(784, [128,64], 10, lr=0.001)
    print(model)
    

    exit(1)
    x_transformed=x.reshape(1,-1)

    for i in tqdm(range(50)):
        out=model.forward(x_transformed)
        y_true=np.zeros((1,10))
        y_true[0,3]=1
        
        loss=model.Loss(y_true, out)
        tqdm.write(f"\u001b[33mSum:\u001b[0m {out.sum(axis=1)[0]:.5f}\t \u001b[33mLoss:\u001b[0m {loss:.4f}")
        # tqdm.write(f"\u001b[33mSum:\u001b[0m {out.sum(axis=1)[0]:.5f}\t \u001b[33mLoss:\u001b[0m {loss:.4f}")

        model.backward(x_transformed, y_true, out)
    

    # plt.imshow(y_true.reshape(1,10))
    # plt.show()