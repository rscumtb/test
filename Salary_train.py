import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import math
import time
import matplotlib.pyplot as plt


class SalaryNet(torch.nn.Module):
    def __init__(self,in_size,h1_size,h2_size,out_size):
        super(SalaryNet,self).__init__()
        self.h1 = torch.nn.Linear(in_size,h1_size)
        self.relu = torch.nn.ReLU()
        self.h2  = torch.nn.Linear(h1_size,h2_size)
        self.out = torch.nn.Linear(h2_size,out_size)

    def forward(self,x):
        h1_relu = self.relu(self.h1(x))
        h2_relu = self.relu(self.h2(h1_relu))
        predict = self.out(h2_relu)
        return predict

def z_score(series):
    _mean = series.sum()/series.count()
    stds = (((series - _mean)**2).sum()/(series.count()-1))**0.5
    new_series = (series-_mean)/stds
    return new_series

if __name__=="__main__":
    df = pd.read_csv("E:\workspace\machin_learning\data\chapter1\datas\salarys.csv", encoding = 'utf8')

    # delete those features not used in this research
    del df['专业']
    del df['省份']
    del df['城市']

    df_xl = pd.get_dummies(df["学历编码"])
    df_zy = pd.get_dummies(df["专业编码"])
    df_wd = pd.get_dummies(df["纬度"])
    df_jd = pd.get_dummies(df["经度"])
    df_sf = pd.get_dummies(df["省份编码"])

    df = pd.concat([df,df_xl,df_zy,df_wd,df_jd,df_sf],axis = 1)

    del df["学历编码"]
    del df["专业编码"]
    del df["纬度"]
    del df["经度"]
    del df["省份编码"]

    target = df["薪酬"]
    del df["薪酬"]
    size = target.count()

    np.random.seed(1314)
    index = np.random.permutation(np.arange(size))
    print(index.size)

    train_X = df.iloc[index[:200000]]
    test_X  = df.iloc[index[200000:]]
    train_Y = target.iloc[index[:200000]]
    test_Y = target.iloc[index[200000:]]

    bTrainMode = False
    #If we need train, change bTrainMode into True,
    #if we want to predict, we can turn bTrainMode into False
    if(bTrainMode):
        # Start training and execute
        batch_size = 512
        epoch = 100
        model = SalaryNet(327, 100, 20, 1)

        #initialize weight
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), 0.001)


        #define a array to visualize the loss
        loss_holder = []
        blocks = math.ceil(train_Y.count() / batch_size)

        loss_value = np.inf

        step = 0

        for i in range(epoch):
            train_count = 0
            baches = 0
            for j in range(blocks):
                train_x_data = torch.Tensor(train_X.iloc[j * batch_size:(j + 1) * batch_size].values)
                train_x_data.requires_grad = True
                train_y_data = torch.Tensor(train_Y.iloc[j * batch_size:(j + 1) * batch_size].values)
                out = model(train_x_data)
                loss = criterion(out.squeeze(1), train_y_data)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(
                    'epoch:{}, Train Loss: {:.6f}, Mean: {:.2f}, Min:{:.2f}, Max:{:.2f}, Median:{:.2f}, Dealed/Records:{}/{}'.format( \
                        i, math.sqrt(loss / batch_size), out.mean(), out.min(), out.max(), out.median(),
                        (j + 1) * batch_size, train_Y.count()))

                if j % 10 == 0:
                    step += 1
                    loss_holder.append([step, math.sqrt(loss / batch_size)])

                if j % 10 == 0 and loss < loss_value:
                    torch.save(model, 'model.ckpt')
                    loss_value = loss
            print("OK")

        fig = plt.figure(figsize=(20, 15))
        fig.autofmt_xdate()
        loss_df = pd.DataFrame(loss_holder, columns=["time", "loss"])
        x_times = loss_df["time"].values
        plt.ylabel("Loss")
        plt.xlabel("times")
        plt.plot(loss_df["loss"].values)
        plt.xticks([10, 100, 400, 700, 1000, 1200, 1500, 2000, 3000, 4000])
        plt.show()

    bPredictMode = True
    if(bPredictMode):
        batch_size = 512
        model_path = 'model.ckpt'
        model = torch.load(model_path)

        model.eval()
        for layer in model.modules():
            layer.requires_grad = False

        criterion = nn.MSELoss()
        results = []
        targets = []
        batches = math.ceil(test_Y.count()/batch_size)

        for i in range(batches):
            test_X_data = torch.Tensor(test_X.iloc[i*batch_size:(i+1)*batch_size].values)
            out         = model(test_X_data)
            target      = torch.Tensor(test_Y.iloc[i*batch_size:(i+1)*batch_size].values)

            if i % 20 == 0:
                results.append(out.squeeze(1))
                targets.append(target)

            loss = criterion(out.squeeze(1),target)
            print("Test Loss: {:.2f}, Mean: {:.2f}, Min: {:.2f}, Max:{:.2f}, Median: {:.2f}, Dealed/Records: {}/{}".format(\
                  math.sqrt(loss/batch_size),out.mean(),out.min(),out.max(),out.median(),(i+1)*batch_size,test_Y.count()))

        results_flatten = []
        targets_flatten = []
        for result in results:
            results_flatten.extend(result.detach().numpy().tolist())

        for target in targets:
            targets_flatten.extend(target.detach().numpy().tolist())

        plt.figure(figsize=(20,10))
        plt.scatter(results_flatten,targets_flatten)
        plt.ylabel("Predict values")
        plt.xlabel("Target values")
        plt.show()

    print("OK")




