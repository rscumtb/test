import torch
import math
import numpy as np
import math
import datetime
import torch.nn as nn
from torchvision import datasets, transforms

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet,self).__init__()
        self.layer1 = nn.Linear(784,50)
        self.layer2 = nn.Linear(50,10)

    def forward(self,x):
        x = self.layer1(x.reshape(-1,784))
        x = torch.tanh(x)

        x = self.layer2(x)
        x = nn.functional.relu(x)
        x = nn.functional.log_softmax(x,dim = 1)
        return x

if __name__ == "__main__":

    start_time = datetime.datetime.now()
    bDownload = True
    if(bDownload):
        mnist_dataset = datasets.MNIST('../data',train = True, download=True, \
                                       transform = transforms.Compose([\
                                           transforms.ToTensor(),\
                                           transforms.Normalize((0.0,),(1.0,))\
                                           ]))
        data_loader = torch.utils.data.DataLoader(mnist_dataset,batch_size = 32, shuffle = True, num_workers = 10,\
                                                  **{"pin_memory": True})

    bTrainMode = True
    if(bTrainMode):
        model = MNISTNet()
        epochs = 5
        train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data',train = True, download=True,\
                                                                  transform=transforms.Compose([transforms.ToTensor,\
                                                                                                transforms.Normalize((0.0,),(1.0,))])),\
                                                   batch_size = 64, shuffle = True, num_workers = 1)
        help(train_loader)


        optimizer = torch.optim.SGD(model.parameters(),0.01,momentum=0.9)

        #Train
        #dataloader_enumerate  = enumerate(data_loader)
        for epoch in range(1,epochs+1):
            model.train()
            for batch_idx, (data, target) in enumerate(data_loader):
                optimizer.zero_grad()
                output = model(data)
                loss   = nn.functional.nll_loss(output, target)
                loss.backward()
                optimizer.step()

                if (batch_idx % 10) == 0:
                    print("Train Epoch: {} [{}/{} ({:.0f})%] Loss: {:.6f}".format(epoch, batch_idx * len(data),len(data_loader.dataset), 100. * batch_idx / len(data_loader), loss.item()))
        #Test
        random_seed = 123
        torch.manual_seed(random_seed)

        test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, \
                                                                  transform=transforms.Compose([transforms.ToTensor, \
                                                                                                transforms.Normalize(
                                                                                                    (0.0,), (1.0,))])), \
                                                   batch_size=64, shuffle=True, num_workers=1)

        #eval
        model.eval()
        test_loss = 0
        correct   = 0
        with torch.no_grad():
            for data, target in data_loader:
                output = model(data)
                test_loss  += nn.functional.nll_loss(output,target,reduction = 'sum').item()
                pred = output.max(1)[1]
                correct += pred.eq(target).sum().item()

        test_loss /= len(data_loader.dataset)
        print("Test Set: Average Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(test_loss,correct,len(data_loader.dataset),100. * correct / len(data_loader.dataset)))


    endtime = datetime.datetime.now()
    print(start_time,endtime)




    print("OK")