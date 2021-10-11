import numpy as np
import scipy.misc
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
import cv2

def sampling(img, num_sam):
    # cvf = np.cumsum(img)  ##cumulation 1Dimg_length
    # cdf = cvf / np.max(cvf)   #normalize
    cdf = np.cumsum(img / np.sum(img))

    # cdf,_,_ = plt.hist(img.ravel(), bins= num_sam,density=True, cumulative=True)

    randomP = np.random.uniform(0,1.,num_sam)
    idx = np.searchsorted(cdf, randomP)   #idx is 1D
    new_img = np.zeros_like(img)
    x = idx // img.shape[1]
    y = idx % img.shape[1]    ##find the idx in 2D_img
    for i in x:
        for j in y:
            new_img[i,j] = img[i,j]
    return new_img

def Parzen_Win(img, kernelWidth):
    kernel = np.ones([kernelWidth,kernelWidth])/kernelWidth**2
    imgNew = scipy.signal.convolve(img, kernel, mode="same")
    return imgNew


if __name__ == '__main__':
    rac = scipy.misc.face(gray=True)
    gau_rac = scipy.ndimage.gaussian_filter(rac, sigma=3)
    plt.gray()
    sam_img = sampling(rac,1000)
    parzen_img = Parzen_Win(rac,9)

    plt.gray()
    plt.subplot(131)
    plt.title("Original Image")
    plt.imshow(gau_rac)
    plt.subplot(132)
    plt.title("Sample Image")
    plt.imshow(sam_img)
    plt.subplot(133)
    plt.title("Parzen Image")
    plt.imshow(parzen_img)
    plt.show()

