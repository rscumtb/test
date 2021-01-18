# This is a sample Python pytorch script.
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


if __name__ == '__main__':
    pytorch  =  np.array(Image.open("..\\data\\chapter1\\imgs\\pytorch.jpg").resize((224,224)))
    print(pytorch.shape)
    pytorch_tensor  =  torch.from_numpy(pytorch)
    pytorch_tensor.size()
    plt.imshow(pytorch)
    print("OK")

