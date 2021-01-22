import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=12,kernel_size=5)

        self.fc1 = nn.Linear(in_features=12*4*4,out_features=120)
        self.fc2 = nn.Linear(in_features=120,out_features=60)
        self.out = nn.Linear(in_features=60,out_features=10)

    def forward(self,t):
        #(1) input layer
        t  = t

        #(2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t,kernel_size = 2, stride = 2)

        #(3) hiddden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t,kernel_size = 2,stride = 2)


        #(4) hiden linear layer
        t = t.reshape(-1,12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        #(5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)


        #(6) hidden linear layer (output)
        t = self.out(t)
        return t


if __name__ == "__main__":

    torch.set_printoptions(linewidth=120)

    train_set = torchvision.datasets.FashionMNIST(
        root = './data/FashionMNIST',
        train = True,
        download=True,
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    )

    torch.set_grad_enabled(True)

    network = Network()

    data_loader = torch.utils.data.DataLoader(
                                                train_set,
                                                batch_size = 100
    )

    optimizer = optim.Adam(network.parameters(), lr=0.01)

    for epoch in range(5):
        total_loss = 0
        total_correct = 0
        for batch in data_loader:
            images, labels = batch
            preds          = network(images)
            loss = F.cross_entropy(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += preds.argmax(dim = 1).eq(labels).sum().item()
        print("Epoch: ",epoch, "Total Loss: ", total_loss, "Total Correct: ", total_correct, "Total Correct Percentage: ", total_correct/len(train_set))




    print("OK")
    print("OK")
    print("OK2")
    print("OK3")
    print("OK4")