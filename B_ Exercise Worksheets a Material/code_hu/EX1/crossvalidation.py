import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

class CrossValidation:
    def __init__(self, k, candidate, groundTruthDistribution, trainFunc):
        self.K = k
        self.kf = KFold(k)
        self.candidates = candidate
        self.gtd = groundTruthDistribution.flatten()
        self.flatten2img = trainFunc[0].reconstract
        self.trainFunc = trainFunc[1].do_sampling


    def calculate_MSE(self,tem,gtd):
        return np.mean((gtd-tem)**2)

    def train(self, gray_idx):
        k = []
        for candidate in self.candidates:
            data = 0
            for train_idx, valid_idx in self.kf.split(gray_idx):
                train_img = self.flatten2img(train_idx)
                recon_img = self.trainFunc(train_img, int(candidate)).flatten()
                recon_flatten = recon_img[valid_idx]
                valid_flatten = self.gtd[valid_idx]
                data += self.calculate_MSE(recon_flatten,valid_flatten)
            k.append(data/self.K)
        return k

    def plotResult(self,train):
        plt.figure("Group 20: Cross Validation")
        plt.suptitle("Group 20: Cross Validation")
        plt.plot(self.candidates,train,"-r") 
        plt.legend(["Error"])
        plt.title("For all sample ")
        plt.xlabel("Window Width")




