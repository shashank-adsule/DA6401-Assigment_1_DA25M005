import numpy as np

class Activation:
    def forward(self, z):
        raise NotImplementedError

    def backward(self, z):
        raise NotImplementedError

    def __call__(self, z):
        return self.forward(z)

class ReLU(Activation):
    def forward(self, z):
        return np.maximum(0, z)

    def backward(self, z):
        return (z > 0).astype(float)


class Sigmoid(Activation):
    def forward(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def backward(self, z):
        s = self.forward(z)
        return s * (1 - s)


class Tanh(Activation):
    def forward(self, z):
        return np.tanh(z)

    def backward(self, z):
        return 1.0 - np.tanh(z) ** 2


class Softmax(Activation):
    def forward(self, z):
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def backward(self, z):
        raise NotImplementedError("Softmax gradient handled in loss")

ACTIVATIONS = {
    "relu": ReLU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "softmax": Softmax,
}

def get_activation(name):
    name = name.lower()
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'")
    return ACTIVATIONS[name]()

if __name__ == "__main__":
    z = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])

    for name, cls in ACTIVATIONS.items():
        if name == "softmax":
            continue
        act = cls()
        print(f"\n{name.upper()}")
        print(f"  forward:  {act.forward(z)}")
        print(f"  backward: {act.backward(z)}")

    print("\nSOFTMAX")
    sm = Softmax()
    out = sm.forward(z)
    print(f"  forward:  {out}")
    print(f"  sum:      {out.sum(axis=1)}  ← should be 1.0")