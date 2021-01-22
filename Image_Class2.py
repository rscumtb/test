import torch
import math
import numpy as np
import math
import datetime
import os
import torch.nn as nn
from torchvision import datasets, transforms
import torch.multiprocessing as mp


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

def train_epoch(epoch,model, device, train_loader, optimizer):
    model.train()
    pid = os.getpid()
    for batch_idx, (data,target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss   = nn.functional.nll_loss(output,target.to(device))
        loss.backward()
        optimizer.step()
        if (batch_idx % 10 ) == 0:
            print("{} Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(pid, epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx/len(train_loader), loss.item()))


def train( model, device, dataloader_pin_memory):
    torch.manual_seed(123)
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', \
                                                              train = True, \
                                                              download=True, \
                                                              transform = transforms.Compose([transforms.ToTensor(),\
                                                                                              transforms.Normalize((0.,),(1.,))])),\
                                               batch_size = 64,\
                                               shuffle    = True,\
                                               num_workers = 1,\
                                               **dataloader_pin_memory)
    optimizer = torch.optim.SGD(model.parameters(),lr = 0.01, momentum = 0.9)
    for epoch in range(1,10 + 1):
        train_epoch(epoch, model, device, train_loader,optimizer)

def test_epoch(model,device, testloader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data,target in testloader:
            output = model(data.to(device))
            test_loss += nn.functional.nll_loss(output,target.to(device),reduction='sum').item()
            pred     = output.max(1)[1]
            correct += pred.eq(target.to(device)).sum().item()
    test_loss /= len(testloader.dataset)
    print("Test set: Average loss: {:.4f}, Accuracy: {}/{}  ({:.0f}%)".format(test_loss, correct, len(testloader.dataset), 100. * correct/len(testloader.dataset)))

def test(model, device, dataloader_pin_memory):
    torch.manual_seed(456)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data',\
                                                             train = False,\
                                                             transform= transforms.Compose([transforms.ToTensor(),\
                                                                                            transforms.Normalize((0.,),(1.,))])),\
                                              batch_size = 64,\
                                              shuffle = True,\
                                              num_workers = 1,\
                                              **dataloader_pin_memory)
    test_epoch(model,device,test_loader)


if __name__ == "__main__":
    starttime = datetime.datetime.now()
    if torch.cuda.is_available():
        device = torch.device("cuda")

    dataloader_pin_memory = {"pin_memory":False}
    #mp.set_start_method("spawn")
    model = MNISTNet().to(device)
    train(model, device, dataloader_pin_memory)
    #model.share_memory()
    #process = []
    #num_process = 10
    #for rank in range(num_process):

        #p = mp.Process(target = train,args = (model, device,dataloader_pin_memory))
        #p.start()
        #process.append(p)
    #for p in process:
        #p.join()
    test(model,device,dataloader_pin_memory)
    endtime = datetime.datetime.now()
    deltatime = endtime - starttime
    print("We have spend :{} seconds.".format(deltatime.seconds))