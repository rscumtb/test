import torch
import math
import torch.nn as nn
import matplotlib.pyplot as plt

def classification(data):
    for i in data:
        if(i[0] > 1.5 + 0.1*torch.rand(1).item()*(-1)**torch.randint(1,10,(1,1)).item()):
            pos.append(i)
        else:
            neg.append(i)

if __name__ == "__main__":
    x1 = torch.randn(365) + 1.5
    x2 = torch.randn(365) - 1.5
    data = zip(x1.data.numpy(),x2.data.numpy())
    pos = []
    neg = []

    classification(data)
    pos_x = [i[0] for i in pos]
    pos_y = [i[1] for i in pos]
    neg_x = [i[0] for i in neg]
    neg_y = [i[1] for i in neg]



