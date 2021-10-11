import numpy as np
import matplotlib.pyplot as plt 
from ex3 import KernelDensity


def generate_data(x, a):
    return np.sin(a*x)

def add_noise(x):
    return np.random.normal(loc=0,scale=0.5,size=np.size(x))+x

if __name__ == "__main__":
    # exercise 1
    x = np.linspace(0,10,100)
    groundTruth = generate_data(x,0.4)
    y = add_noise(groundTruth)
    # show exercise 1
    plt.figure("Group 20: Result Image")
    plt.subplot(221)
    plt.plot(x,groundTruth,"b")
    plt.scatter(x,y,color='', marker='o', edgecolors='c')

    # exercise 2
    kde = KernelDensity(x,y,bw=2.0)
    epan = np.zeros_like(y)
    lolr = np.zeros_like(y)

    for i in range(np.size(x)):
        epan[i] = kde.epanechnikov(x[i])
        lolr[i] = kde.localLinearRegression(x[i])
    plt.plot(x, epan, "g")
    plt.plot(x, lolr, "r")
    plt.title("Ground Truth = sin(0.4x) with gaussian noisy(mean=0,var=0.5)")
    plt.legend(["Ground truth","Epanechnikov","local linear regression",
        "Noisy samples"],loc = 'upper right')
    

    
    plt.subplot(222)
    plt.plot(x,groundTruth,"b")
    plt.scatter(x,y,color='', marker='o', edgecolors='c')
    pos = 1
    _ = kde.epanechnikov(x[pos])
    kEpan = kde.getKernel()
    plt.plot(x,kEpan,"y")    
    _ = kde.localLinearRegression(x[pos])
    kLolr = kde.getKernel()
    plt.plot(x,kLolr,"g")
    plt.plot(x,kLolr*10,"r")     
    plt.title("Kernel Function with x0 = {}".format(pos))
    plt.legend(["Ground truth","Epanechnikov","local linear regression",
        "local linear regression * 10","Noisy samples"],loc = 'upper right')
    

    
    plt.subplot(223)
    plt.plot(x,groundTruth,"b")
    plt.scatter(x,y,color='', marker='o', edgecolors='c')
    pos = 25
    _ = kde.epanechnikov(x[pos])
    kEpan = kde.getKernel()
    plt.plot(x,kEpan,"y")    
    _ = kde.localLinearRegression(x[pos])
    kLolr = kde.getKernel()
    plt.plot(x,kLolr,"g")
    plt.plot(x,kLolr*10,"r")    
    plt.title("Kernel Function with x0 = {}".format(pos))
    plt.legend(["Ground truth","Epanechnikov","local linear regression",
        "local linear regression * 10","Noisy samples"],loc = 'upper right')

    plt.subplot(224)
    plt.plot(x,groundTruth,"b")
    plt.scatter(x,y,color='', marker='o', edgecolors='c')
    pos = 99
    _ = kde.epanechnikov(x[pos])
    kEpan = kde.getKernel()
    plt.plot(x,kEpan,"y")    
    _ = kde.localLinearRegression(x[pos])
    kLolr = kde.getKernel()
    plt.plot(x,kLolr,"g")
    plt.plot(x,kLolr*10,"r")    
    plt.title("Kernel Function with x0 = {}".format(pos))
    plt.legend(["Ground truth","Epanechnikov","local linear regression",
        "local linear regression * 10","Noisy samples"],loc = 'upper right')
    


    plt.suptitle("Sample Distributions with bandwidth = {}, sample number = {}".format(2.0,100))
    
    plt.show()