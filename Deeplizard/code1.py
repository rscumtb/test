import torch
import torch.nn as nn
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

    train_loader = torch.utils.data.DataLoader(train_set,batch_size = 10)
    print("length of train set is {}".format(len(train_set)))
    print(train_set.targets)
    print(train_set.targets.bincount())

    sample = next(iter(train_set))
    image,label = sample
    print(image.shape)
    plt.imshow(image.squeeze())


    train_loader = torch.utils.data.DataLoader(train_set,batch_size = 100)
    batch = next(iter(train_loader))
    images,labels = batch
    print(image.shape)
    grid = torchvision.utils.make_grid(images,nrow = 10)
    print("Grid shape is {}".format(grid.shape))
    plt.imshow(np.transpose(grid,(1,2,0)))
    print("Lables: ", labels)



    print("OK")
    print("OK2")

