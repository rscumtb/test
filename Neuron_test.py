import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression,self).__init__()
        self.linear = torch.nn.Linear(1,1)
    def forward(self,x):
        return self.linear(x)


if __name__ == "__main__":
    x = np.arange(20)
    y = np.array([5*x[i] + random.randint(1,20) for i in range(len(x))])


    x_train = torch.from_numpy(x).float()
    y_train = torch.from_numpy(y).float()

    model   =  LinearRegression()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(),0.001)

    num_epochs = 10
    for i in range(num_epochs):
        input_data = x_train.unsqueeze(1)
        target     = y_train.unsqueeze(1)

        out = model(input_data)
        loss = criterion(out,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Epoch:[{}/{}], Loss:[{: .4f}]".format(i+1,num_epochs,loss.item()))

        if((i+1)%2==0):
            predict  =  model(input_data)
            plt.plot(x_train.data.numpy(),predict.squeeze(1).data.numpy(),"r")
            loss2 = criterion(predict,target)
            plt.title("Loss: {:.4f}".format(loss2.item()))
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.scatter(x_train,y_train)
            plt.show()
            print("OK")



    print("OK")
