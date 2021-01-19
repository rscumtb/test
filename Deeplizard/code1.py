import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt



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
    print(train_set.train_labels)
    print(train_set.train_labels.bincount())

    sample = next(iter(train_set))
    image,label = sample

    print(image.shape)

    plt.imshow(image.squeeze())

    print("OK")
