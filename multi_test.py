import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import os

def foo(q):
    pid = os.getpid()
    q.put('my pid is: {}'.format(pid))
    print(pid)

if  __name__ == "__main__":
    mp.set_start_method('spawn')

    q = mp.Queue()
    ps = []

    for i in range(10):
        ps.append(mp.Process(target = foo,args = (q,)))
    for p in ps:
        p.start()

    for p in ps:
        p.join()

    data = q.get()
    while(data):
        print(data)
        data=q.get()