import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

# use same number as seed to make sure all group members become same Image
SEEDNUM = 1
def gaussian1d(x,mu,sigma):
    return 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-1 * (x-mu)**2/(2*sigma**2))

def gaussian2d(x,y,mu,sigma):
    sigmaX = sigma[0,0]
    sigmaY = sigma[1,1]
    muX = mu[0]
    muY = mu[1]
    front = 1/(sigmaX*sigmaY*(np.sqrt(2*np.pi))**2)
    fac = (x-muX)**2/sigmaX**2 + (y-muY)**2/sigmaY**2
    return front * np.exp(-1/2*fac)

def exercise_1(sample,binNum):
    """ Exercise 1: 1d gaussian distribution  """
    # Given information：
    sampleNum = sample
    bins = binNum
    mean = 1
    deviation = 0.2

    # Experimental
    # generate random data with gaussian distribution
    np.random.seed(SEEDNUM)
    allRandomData = np.random.normal(mean, deviation, sampleNum )
    _,x_array = np.histogram(allRandomData,bins,  normed=True)
    
    # Ground Truth
    x = np.linspace(np.floor(min(x_array)),np.ceil(max(x_array)),sampleNum)
    y = gaussian1d(x,mean,deviation)
    # show image
    # plt.figure("Experimental and Ground Truth Distribution in 1D ", figsize=(10,5))
    plt.figure()
    plt.subplot(1,2,1)
    plt.hist(allRandomData,bins,  normed=True)
    plt.title("Experimental in 2D")
    plt.subplot(1,2,2)
    plt.plot(x,y,"r")
    plt.title("2D Distribution")

    plt.suptitle("Gaussian with bins = {}, sample = {}".format(bins,sampleNum) + "\n"
                         + "($\\mu = {},\\sigma = {}$)".format(mean,deviation))
    

def exercise_2(sample,binNum):
    """ Exercise 2: 2d gaussian distribution  """
    # Given information：
    sampleNum = sample
    mean = np.array([0.5,-0.2])
    deviation =  np.array([[2.0,0.3],[0.3,0.5]])
    bins =  np.array([binNum,binNum])

    # Experimental
    # generate random data with gaussian distribution
    np.random.seed(SEEDNUM)
    allRandomData = np.random.multivariate_normal(mean, deviation, sampleNum )
    hist = np.histogram2d(allRandomData[:,0],allRandomData[:,1],bins=bins, normed=True)
    zBar = hist[0].flatten()
    xEdges = hist[1][:-1]
    yEdges = hist[2][:-1]
    x, y = np.meshgrid(xEdges, yEdges)
    xPos = x.flatten()
    yPos = y.flatten()

    # # Ground Truth
    z = gaussian2d(x,y,mean,deviation)
    # show image
    # fig = plt.figure("Experimental and Ground Truth Distribution in 2D ", figsize=(10,5))
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax1.bar3d(xPos, yPos, 0, 1, 1, zBar)
    plt.title("Experimental in 3D")

    ax2 = fig.add_subplot(1,2,2, projection='3d')
    ax2.plot_wireframe(x, y, z, color="r"   )
    plt.title("3D Distribution")
    plt.suptitle("Gaussian with bins = {}, sample = {}".format(bins,sampleNum) + "\n"
                         + "($\\mu = {},\\sigma = {})$".format(mean,deviation))


if __name__ == "__main__":
    # # ex 1
    # exercise_1(1000,30)
    # # ex 2
    # exercise_2(10000,30)

    """
    In the case of constant bin, as the sampling rate gradually increases, the closer the obtained image is to the real image

    When the sampling rate is constant, the image is gradually distorted as the bin gradually increases.

    When both are close to infinity, the image is closest to the real image
    """
    # # ex 3    
    exercise_1(20000,10)    
    exercise_1(20000,30)    
    exercise_1(20000,100)    
    # exercise_1(100000,30)    
    # exercise_1(1000,300)
    plt.show()