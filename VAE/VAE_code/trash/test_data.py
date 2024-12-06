import os
import numpy as np

fd = open(os.path.join('../../dataset/MNIST','train-images-idx3-ubyte'))
loaded = np.fromfile(fd, dtype=np.uint8)
trX = loaded[16:].reshape((60000, 28, 28)).astype(float)
# trX = (trX - 128.0) / 255.0
trX = trX / 255.0
result = trX[0, :, :]

print(result)