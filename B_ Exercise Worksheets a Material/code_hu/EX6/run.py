import numpy as np

from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_swiss_roll

from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.manifold import SpectralEmbedding as LE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



SHOW_DATASET = True

DO_PCA = True
SHOW_PCA = True

DO_MDS = True
SHOW_MDS = True

DO_IM = True
SHOW_IM = True

DO_LE = True
SHOW_LE = True


def print1D(data):
    plt.scatter(data[0][:,0],data[1],s=3,marker='o')

def print2D(data):
    plt.scatter(data[0][:,0],data[0][:,1],c=data[1],s=3,marker='o')

def print3D(data,ax):
    ax.scatter(data[0][:,0],data[0][:,1],data[0][:,2],c=data[1],s=3,marker='o')


SAMPLE_NUMBER = 500
OneGaussian2D = make_blobs(n_samples=SAMPLE_NUMBER,n_features=2,
                                centers=[[1,1]],
                                cluster_std=[0.5])

MultiGaussian2D = make_blobs(n_samples=SAMPLE_NUMBER,n_features=2,
                                centers=[[1,1],[5,5],[-1,1],[3,4]],
                                cluster_std=[0.5,0.1,0.3,0.8])

MultiGaussian3D = make_blobs(n_samples=SAMPLE_NUMBER,n_features=3,
                                centers=[[1,1,1],[5,5,5],[-1,8,1],[3,2,4]],
                                cluster_std=[0.5,0.1,0.3,0.8])
TwoCircles2D = make_circles(n_samples=SAMPLE_NUMBER,shuffle=False,
                                noise=.075,
                                factor=.5)

SwissRoll3D = make_swiss_roll(n_samples=SAMPLE_NUMBER,noise=0.075)

if SHOW_DATASET:
    plt.figure("Dataset")
    plt.subplot(231)
    print2D(OneGaussian2D)
    plt.title("One Gaussian Distribution 2D")
    plt.subplot(232)
    print2D(MultiGaussian2D)
    plt.title("Multiple Gaussian Distribution 2D")
    ax = plt.subplot(233,projection='3d')
    print3D(MultiGaussian3D,ax)
    plt.title("Multiple Gaussian Distribution 3D")
    plt.subplot(234)
    print2D(TwoCircles2D)
    plt.title("Two circles Distribution 2D")
    ax = plt.subplot(235,projection='3d')
    print3D(SwissRoll3D,ax)
    plt.title("Swiss roll Distribution 2D")

    plt.suptitle('Distributions in Original')

if DO_PCA:
    pca2D = PCA(n_components=2)
    pca3D = PCA(n_components=3)

    OG_pca = pca2D.fit_transform(OneGaussian2D[0])
    MG_pca = pca2D.fit_transform(MultiGaussian2D[0])
    MG_pca3 = pca3D.fit_transform(MultiGaussian3D[0])
    CC_pca = pca2D.fit_transform(TwoCircles2D[0])
    SR_pca3 = pca3D.fit_transform(SwissRoll3D[0])    

    if SHOW_PCA:        
        plt.figure("Dataset after PCA")
        plt.subplot(231)
        print1D([OG_pca,OneGaussian2D[1]])
        print1D([OG_pca,-1*np.ones_like(OneGaussian2D[1])])
        plt.legend(["dist with label","real dist"])
        plt.title("One Gaussian Distribution 2D")
        
        plt.subplot(232)
        print1D([MG_pca,MultiGaussian2D[1]])
        print1D([MG_pca,-1*np.ones_like(MultiGaussian2D[1])])
        plt.legend(["dist with label","real dist"])
        plt.title("Multiple Gaussian Distribution 2D")
        
        plt.subplot(233)
        print2D([MG_pca3,MultiGaussian3D[1]])
        plt.title("Multiple Gaussian Distribution 3D")

        plt.subplot(234)
        print1D([CC_pca,TwoCircles2D[1]])
        print1D([CC_pca,-1*np.ones_like(TwoCircles2D[1])])
        plt.legend(["dist with label","real dist"])
        plt.title("Two circles Distribution 2D")

        plt.subplot(235)
        print2D([SR_pca3,SwissRoll3D[1]])
        plt.title("Swiss roll Distribution 2D")

        plt.suptitle('Distributions after PCA')

if DO_MDS:
    mds2D = MDS(n_components=2)
    mds3D = MDS(n_components=3)

    OG_mds = mds2D.fit_transform(OneGaussian2D[0])
    MG_mds = mds2D.fit_transform(MultiGaussian2D[0])
    MG_mds3 = mds3D.fit_transform(MultiGaussian3D[0])
    CC_mds = mds2D.fit_transform(TwoCircles2D[0])
    SR_mds3 = mds3D.fit_transform(SwissRoll3D[0]) 

    if SHOW_MDS:        
        plt.figure("Dataset after MDS")
        plt.subplot(231)
        print1D([OG_mds,OneGaussian2D[1]])
        print1D([OG_mds,-1*np.ones_like(OneGaussian2D[1])])
        plt.legend(["dist with label","real dist"])
        plt.title("One Gaussian Distribution 2D")
        
        plt.subplot(232)
        print1D([MG_mds,MultiGaussian2D[1]])
        print1D([MG_mds,-1*np.ones_like(MultiGaussian2D[1])])
        plt.legend(["dist with label","real dist"])
        plt.title("Multiple Gaussian Distribution 2D")
        
        plt.subplot(233)
        print2D([MG_mds3,MultiGaussian3D[1]])
        plt.title("Multiple Gaussian Distribution 3D")

        plt.subplot(234)
        print1D([CC_mds,TwoCircles2D[1]])
        print1D([CC_mds,-1*np.ones_like(TwoCircles2D[1])])
        plt.legend(["dist with label","real dist"])
        plt.title("Two circles Distribution 2D")

        plt.subplot(235)
        print2D([SR_mds3,SwissRoll3D[1]])
        plt.title("Swiss roll Distribution 2D")

        plt.suptitle('Distributions after MDS')

if DO_IM:
    isomap2D = Isomap(n_components=2)
    isomap3D = Isomap(n_components=3)

    OG_isomap = isomap2D.fit_transform(OneGaussian2D[0])
    MG_isomap = isomap2D.fit_transform(MultiGaussian2D[0])
    MG_isomap3 = isomap3D.fit_transform(MultiGaussian3D[0])
    CC_isomap = isomap2D.fit_transform(TwoCircles2D[0])
    SR_isomap3 = isomap3D.fit_transform(SwissRoll3D[0]) 

    if SHOW_IM:        
        plt.figure("Dataset after Iso Map")
        plt.subplot(231)
        print1D([OG_isomap,OneGaussian2D[1]])
        print1D([OG_isomap,-1*np.ones_like(OneGaussian2D[1])])
        plt.legend(["dist with label","real dist"])
        plt.title("One Gaussian Distribution 2D")
        
        plt.subplot(232)
        print1D([MG_isomap,MultiGaussian2D[1]])
        print1D([MG_isomap,-1*np.ones_like(MultiGaussian2D[1])])
        plt.legend(["dist with label","real dist"])
        plt.title("Multiple Gaussian Distribution 2D")
        
        plt.subplot(233)
        print2D([MG_isomap3,MultiGaussian3D[1]])
        plt.title("Multiple Gaussian Distribution 3D")

        plt.subplot(234)
        print1D([CC_isomap,TwoCircles2D[1]])
        print1D([CC_isomap,-1*np.ones_like(TwoCircles2D[1])])
        plt.legend(["dist with label","real dist"])
        plt.title("Two circles Distribution 2D")

        plt.subplot(235)
        print2D([SR_isomap3,SwissRoll3D[1]])
        plt.title("Swiss roll Distribution 2D")

        plt.suptitle('Distributions after Iso Map')

if DO_LE:
    lmap2D = LE(n_components=2)
    lmap3D = LE(n_components=3)

    OG_lmap = lmap2D.fit_transform(OneGaussian2D[0])
    MG_lmap = lmap2D.fit_transform(MultiGaussian2D[0])
    MG_lmap3 = lmap3D.fit_transform(MultiGaussian3D[0])
    CC_lmap = lmap2D.fit_transform(TwoCircles2D[0])
    SR_lmap3 = lmap3D.fit_transform(SwissRoll3D[0]) 

    if SHOW_LE:        
        plt.figure("Dataset after Laplacian Eigenmaps")
        plt.subplot(231)
        print1D([OG_lmap,OneGaussian2D[1]])
        print1D([OG_lmap,-1*np.ones_like(OneGaussian2D[1])])
        plt.legend(["dist with label","real dist"])
        plt.title("One Gaussian Distribution 2D")
        
        plt.subplot(232)
        print1D([MG_lmap,MultiGaussian2D[1]])
        print1D([MG_lmap,-1*np.ones_like(MultiGaussian2D[1])])
        plt.legend(["dist with label","real dist"])
        plt.title("Multiple Gaussian Distribution 2D")
        
        plt.subplot(233)
        print2D([MG_lmap3,MultiGaussian3D[1]])
        plt.title("Multiple Gaussian Distribution 3D")

        plt.subplot(234)
        print1D([CC_lmap,TwoCircles2D[1]])
        print1D([CC_lmap,-1*np.ones_like(TwoCircles2D[1])])
        plt.legend(["dist with label","real dist"])
        plt.title("Two circles Distribution 2D")

        plt.subplot(235)
        print2D([SR_lmap3,SwissRoll3D[1]])
        plt.title("Swiss roll Distribution 2D")

        plt.suptitle('Distributions after Laplacian Eigenmaps')

plt.show()








