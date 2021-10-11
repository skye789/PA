import numpy as np

import scipy.misc
import scipy.signal
import scipy.ndimage
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
from sklearn.model_selection import KFold

SHOW_IMG = False
SHOW_GAU = False
SHOW_CDF = False
SHOW_EX1 = False

SHOW_EX2 = False

class Exercise_1:
    def __init__(self,SIGEMA):        
        self.img = scipy.misc.face(gray=True)
        self.img_gaussian = scipy.ndimage.gaussian_filter(self.img,SIGEMA)
        if SHOW_IMG:
            plt.figure("Original Image")
            plt.gray()
            plt.imshow(self.img)
        if SHOW_GAU:
            plt.figure("Gaussian Image")
            plt.gray()
            plt.imshow(self.img_gaussian)
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
        imgNew[x,y] = 1
        if SHOW_EX1:
            plt.figure("New Image with sample = {}".format(sampleNum))
            plt.imshow(imgNew)
        return imgNew

    def get_img(self):
        return self.img

class Exercise_2:
    def __init__(self):
        pass
    
    def do_sampling(self, img, kernelWidth):
        kernel = np.ones([kernelWidth,kernelWidth])/kernelWidth**2
        imgNew = scipy.signal.convolve(img, kernel, mode="same")
        if SHOW_EX2:            
            plt.figure("New Image with kernel width = {}".format(kernelWidth))
            plt.imshow(imgNew)
        return imgNew


class CrossValidation:
    def __init__(self, k, candidate, groundTruthDistribution, trainFunc):
        self.K = k
        self.kf = KFold(k)
        self.candidates = candidate
        self.gtd = groundTruthDistribution
        self.flatten2img = trainFunc[0].reconstract
        self.trainFunc = trainFunc[1].do_sampling

    def train(self, gray_idx):
        k = []
        for candidate in self.candidates:
            theta = 0
            for train_idx, valid_idx in self.kf.split(gray_idx):
                # translate the index
                train_idx = gray_idx[train_idx]
                valid_idx = gray_idx[valid_idx]
                # recreate img then reconstraction
                train_img = self.flatten2img(train_idx)
                recon_img = self.trainFunc(train_img, int(candidate))
                
                train_pdf = recon_img.flatten() + 1e-8
                valid_sample = train_pdf[valid_idx]
                theta += np.sum(-np.ma.log(valid_sample/np.sum(train_pdf)))
            k.append(theta/self.K)

            # plt.figure("train_img {}".format(candidate))
            # plt.imshow(train_img)
            # plt.figure("recon_img {}".format(candidate))
            # plt.imshow(recon_img)
            # # plt.show()
        return np.array(k)

    def plotResult(self,res,sample):
        
        idx = np.where(res==np.min(res))
        opt = int(self.candidates[idx])

        plt.figure("Group 20: Cross Validation with sample size {}".format(sample))
        plt.suptitle("Group 20: Cross Validation with sample size {}".format(sample))
        plt.plot(self.candidates,res,"-b") 
        plt.plot(self.candidates[idx],res[idx],"*r")
        plt.legend(["theta","optimal"])
        plt.title("Optimal theta with kernel size = {}".format(opt))
        plt.xlabel("Window Width")
        plt.ylabel("theta")