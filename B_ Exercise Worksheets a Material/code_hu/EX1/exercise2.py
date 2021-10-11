import numpy as np
import scipy.signal
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt

SHOW_EX2 = False

class Exercise_2:
    def __init__(self):
        pass
    
    def do_sampling(self, img, kernelWidth):
        kernel = np.ones([kernelWidth,kernelWidth])
        imgNew = convolve(img,kernel)
        if SHOW_EX2:            
            plt.figure("New Image with kernel width = {}".format(kernelWidth))
            plt.imshow(imgNew)
        return imgNew