import scipy.misc
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt

SHOW_IMG = False
SHOW_CDF = False
SHOW_EX1 = False

class Exercise_1:
    def __init__(self,SIGEMA):        
        self.img = scipy.misc.face(gray=True)
        self.img = scipy.ndimage.gaussian_filter(self.img,SIGEMA)
        if SHOW_IMG:
            plt.figure("Original Image")
            plt.gray()
            plt.imshow(self.img)
        cvf = np.cumsum(self.img)
        self.cdf = cvf/np.max(cvf)
        if SHOW_CDF:
            plt.figure("CDF")
            plt.plot(self.cdf)

    def do_sampling(self, sampleNum):
        idx = self.get_idx(sampleNum)
        imgNew = self.reconstract(idx)
        return imgNew
        
    def get_idx(self,sampleNum):        
        randomP = np.random.uniform(0,1.,sampleNum)
        idx = np.searchsorted(self.cdf, randomP)
        return idx
    
    def reconstract(self,idx):         
        imgNew = np.zeros_like(self.img)       
        x = idx // self.img.shape[1]
        y = idx % self.img.shape[1]
        imgNew[x,y] = self.img[x,y]
        if SHOW_EX1:
            plt.figure("New Image with sample = {}".format(sampleNum))
            plt.imshow(imgNew)
        return imgNew

    def get_img(self):
        return self.img

