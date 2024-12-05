'''Load data from dataset !!!'''
################################################
import os
import numpy as np
from torchvision import datasets
################################################


# MNIST dataset:
#   60000 training samples and 10000 test samples
#   each figs: 3 * 28 * 28 
def load_data_mnist(data_dir):
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28)).astype(float)
    trX = np.expand_dims(trX, axis=1)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000))

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28)).astype(float)
    teX = np.expand_dims(teX, axis=1)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000))

    trX = (trX - 128.0) / 255.0
    teX = (teX - 128.0) / 255.0

    return trX, trY, teX, teY


# CIFAR-10 dataset:
#   60000 figs from 10 types of objects
#   each figs: 3 * 32 * 32