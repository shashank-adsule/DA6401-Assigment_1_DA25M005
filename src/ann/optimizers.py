import numpy as np

class Optimizer:
    def __init__(self, lr):
        self.lr = lr
        self.initialized = False

    def _init_state(self, layers):
        raise NotImplementedError

    def step(self, layers):
        raise NotImplementedError

# -------------------- SGD --------------------
class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__(lr)

    def _init_state(self, layers):
        self.initialized = True

    def step(self, layers):
        if not self.initialized:
            self._init_state(layers)

        for layer in layers:
            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b

# -------------------- Momentum --------------------
class Momentum(Optimizer):
    def __init__(self, lr=0.01, beta=0.9):
        super().__init__(lr)
        self.beta = beta

    def _init_state(self, layers):
        self.mW = [np.zeros_like(l.W) for l in layers]
        self.mb = [np.zeros_like(l.b) for l in layers]
        self.initialized = True

    def step(self, layers):
        if not self.initialized:
            self._init_state(layers)

        for i, layer in enumerate(layers):
            self.mW[i] = self.beta * self.mW[i] + layer.grad_W
            self.mb[i] = self.beta * self.mb[i] + layer.grad_b

            layer.W -= self.lr * self.mW[i]
            layer.b -= self.lr * self.mb[i]

# -------------------- NAG --------------------
class NAG(Optimizer):
    def __init__(self, lr=0.01, beta=0.9):
        super().__init__(lr)
        self.beta = beta

    def _init_state(self, layers):
        self.vW = [np.zeros_like(l.W) for l in layers]
        self.vb = [np.zeros_like(l.b) for l in layers]
        self.initialized = True

    def step(self, layers):
        if not self.initialized:
            self._init_state(layers)

        for i, layer in enumerate(layers):
            self.vW[i] = self.beta * self.vW[i] + layer.grad_W
            self.vb[i] = self.beta * self.vb[i] + layer.grad_b

            layer.W -= self.lr * (self.beta * self.vW[i] + layer.grad_W)
            layer.b -= self.lr * (self.beta * self.vb[i] + layer.grad_b)

# -------------------- RMSProp --------------------
class RMSProp(Optimizer):
    def __init__(self, lr=0.001, beta=0.9, eps=1e-8):
        super().__init__(lr)
        self.beta = beta
        self.eps = eps

    def _init_state(self, layers):
        self.vW = [np.zeros_like(l.W) for l in layers]
        self.vb = [np.zeros_like(l.b) for l in layers]
        self.initialized = True

    def step(self, layers):
        if not self.initialized:
            self._init_state(layers)

        for i, layer in enumerate(layers):
            self.vW[i] = self.beta * self.vW[i] + (1 - self.beta) * layer.grad_W**2
            self.vb[i] = self.beta * self.vb[i] + (1 - self.beta) * layer.grad_b**2

            layer.W -= self.lr * layer.grad_W / (np.sqrt(self.vW[i]) + self.eps)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(self.vb[i]) + self.eps)

# -------------------- Adam --------------------
class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

    def _init_state(self, layers):
        self.mW = [np.zeros_like(l.W) for l in layers]
        self.mb = [np.zeros_like(l.b) for l in layers]
        self.vW = [np.zeros_like(l.W) for l in layers]
        self.vb = [np.zeros_like(l.b) for l in layers]
        self.initialized = True

    def step(self, layers):
        if not self.initialized:
            self._init_state(layers)

        self.t += 1

        for i, layer in enumerate(layers):
            self.mW[i] = self.beta1 * self.mW[i] + (1 - self.beta1) * layer.grad_W
            self.mb[i] = self.beta1 * self.mb[i] + (1 - self.beta1) * layer.grad_b

            self.vW[i] = self.beta2 * self.vW[i] + (1 - self.beta2) * layer.grad_W**2
            self.vb[i] = self.beta2 * self.vb[i] + (1 - self.beta2) * layer.grad_b**2

            mW_hat = self.mW[i] / (1 - self.beta1**self.t)
            mb_hat = self.mb[i] / (1 - self.beta1**self.t)
            vW_hat = self.vW[i] / (1 - self.beta2**self.t)
            vb_hat = self.vb[i] / (1 - self.beta2**self.t)

            layer.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
            layer.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)

# -------------------- Nadam --------------------
class Nadam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

    def _init_state(self, layers):
        self.mW = [np.zeros_like(l.W) for l in layers]
        self.mb = [np.zeros_like(l.b) for l in layers]
        self.vW = [np.zeros_like(l.W) for l in layers]
        self.vb = [np.zeros_like(l.b) for l in layers]
        self.initialized = True

    def step(self, layers):
        if not self.initialized:
            self._init_state(layers)

        self.t += 1

        for i, layer in enumerate(layers):
            self.mW[i] = self.beta1 * self.mW[i] + (1 - self.beta1) * layer.grad_W
            self.mb[i] = self.beta1 * self.mb[i] + (1 - self.beta1) * layer.grad_b

            self.vW[i] = self.beta2 * self.vW[i] + (1 - self.beta2) * layer.grad_W**2
            self.vb[i] = self.beta2 * self.vb[i] + (1 - self.beta2) * layer.grad_b**2

            mW_hat = self.mW[i] / (1 - self.beta1**self.t)
            mb_hat = self.mb[i] / (1 - self.beta1**self.t)
            vW_hat = self.vW[i] / (1 - self.beta2**self.t)
            vb_hat = self.vb[i] / (1 - self.beta2**self.t)

            nW = self.beta1 * mW_hat + (1 - self.beta1) / (1 - self.beta1**self.t) * layer.grad_W
            nb = self.beta1 * mb_hat + (1 - self.beta1) / (1 - self.beta1**self.t) * layer.grad_b

            layer.W -= self.lr * nW / (np.sqrt(vW_hat) + self.eps)
            layer.b -= self.lr * nb / (np.sqrt(vb_hat) + self.eps)

OPTIMIZERS = {
    "sgd": SGD,
    "momentum": Momentum,
    "nag": NAG,
    "rmsprop": RMSProp,
    "adam": Adam,
    "nadam": Nadam,
}


def get_optimizer(name, lr, **kwargs):
    name = name.lower()
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer '{name}'")
    return OPTIMIZERS[name](lr=lr, **kwargs)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/claude/da6401_assignment1")
    from layers import Linear
    from losses import CrossEntropyLoss

    np.random.seed(42)
    x = np.random.randn(32, 784)
    y = np.zeros((32, 10))
    y[np.arange(32), np.random.randint(0, 10, 32)] = 1

    loss_fn = CrossEntropyLoss()

    print("Testing all optimizers for 5 steps each:\n")
    for opt_name in OPTIMIZERS:
        # Fresh layers for each optimizer
        layers = [
            Linear(784, 128, activation="relu",    init="xavier"),
            Linear(128, 10,  activation="softmax", init="xavier"),
        ]
        opt = get_optimizer(opt_name, lr=0.001)
        losses = []
        for step in range(5):
            a = x
            for layer in layers:
                a = layer.forward(a)
            l = loss_fn(y, a)
            losses.append(l)
            delta = loss_fn.backward(y, a)
            for layer in reversed(layers):
                delta = layer.backward(delta)
            opt.step(layers)

        print(f"  {opt_name:10s}: loss {losses[0]:.4f} → {losses[-1]:.4f}")