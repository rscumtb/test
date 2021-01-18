import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math





if __name__ == "__main__":
    x = torch.randn(2,5, requires_grad = True)
    y = torch.sin(x)
    weights_holder = torch.ones(2,5)
    y.backward(weights_holder)
    print(x.grad)
