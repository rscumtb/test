import torch
import torch.nn as nn
import pandas as pd



if __name__ == "__main__":
    df   =  pd.read_csv(r"E:\workspace\machin_learning\data\chapter2\Titanic-dataset\train.csv",encoding = 'utf8')
    df.groupby("Survived").count()
    print(df)
    data1 = torch.randint(10,100,(2,100))
    print("OK")