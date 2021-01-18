import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

##f(x) = -1.13x -2.14x^2 + 3.15x^3 -0.01x^4 +0.512


def features(x):
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1,5)],1)

def target(x):
    return x.mm(x_weights)+b.item()

def get_batch_data(batch_size):
    batch_x = torch.randn(batch_size)
    print(batch_x.shape)
    features_x = features(batch_x)
    print(features_x.shape)
    target_y   = target(features_x)
    return features_x, target_y

class PolynomiaRegression(torch.nn.Module):
    def __init__(self):
        super(PolynomiaRegression,self).__init__()
        self.poly = torch.nn.Linear(4,1)
    def forward(self,x):
        return self.poly(x)

if __name__ == "__main__":
    x = torch.linspace(-2,2,50)
    y = -1.13*x - 2.14*torch.pow(x,2) + 3.15*torch.pow(x,3) - 0.01*torch.pow(x,4) + 0.512

    x_weights  = torch.Tensor([-1.13, -2.14, 3.15, - 0.01]).unsqueeze(1)
    b          = torch.Tensor([0.512])

    epoches = 10000
    batch_size = 32
    model = PolynomiaRegression()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(),0.001)

    for epoch in range(epoches):
        batch_x, batch_y = get_batch_data(batch_size)
        out = model(batch_x)
        loss  = criterion(out,batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch % 100 == 0):
            print("Epoch: [{}/{}], loss: [{:.6f}]".format(epoch + 1, epoches,loss.item()))

            if(epoch % 1000 == 0):
                predict = model(features(x))
                plt.plot(x.data.numpy(),predict.squeeze(1).data.numpy(),"r")
                loss2 = criterion(predict,y)
                plt.title = ("Loss: {: .4f}".format(loss.item()))
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.scatter(x,y)
                plt.show()
                print("OK")


    print("OK")