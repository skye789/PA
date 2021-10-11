import numpy as np
from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans

from KMEANS import KMeans_Self

import matplotlib.pyplot as plt

def create_dataset():
    cood0 = np.random.normal(size=(2,100))
    cood1 = np.random.normal(size=(2,100))
    cood2 = np.random.normal(size=(2,100))

    center0 = (5.0,7.5)
    center1 = (2.5,3.5)
    center2 = (8.0,4.0)

    
    cood0[0] = cood0[0] + center0[0]
    cood0[1] = cood0[1] + center0[1]
    cood1[0] = cood1[0] + center1[0]
    cood1[1] = cood1[1] + center1[1]
    cood2[0] = cood2[0] + center2[0]
    cood2[1] = cood2[1] + center2[1]

    plt.scatter(cood0[0],cood0[1],c='r')
    plt.scatter(cood1[0],cood1[1],c='g')
    plt.scatter(cood2[0],cood2[1],c='b')
    plt.title("ground trough")

    result = np.concatenate((cood0,cood1),axis=1)
    result = np.concatenate((result,cood2),axis=1)
    return result

def kmean_forbidden(data,k):
    color = ['r','g','b','y','c','m','k','gold','cyan','purple','magenta']
    res = KMeans(n_clusters=k).fit(data)
    labels = res.labels_
    for label,cood in zip(labels,data):
        plt.scatter(cood[0],cood[1],c=color[int(label)])
    plt.title("kmeans with Scipy \n K = {}".format(k))

def meanShift_forbidden(data,bw):    
    color = ['r','g','b','y','c','m','k','gold','cyan','purple','magenta']
    res = MeanShift(bandwidth=bw).fit(data)
    labels = res.labels_
    # print(labels)
    for label,cood in zip(labels,data):
        plt.scatter(cood[0],cood[1],c=color[int(label)])
    plt.title("MeanShift with Scipy \n BandWidth = {}".format(bw))

# def kmean_self(data,k):
#     color = ['r','g','b','y','c','m','k','gold','cyan','purple','magenta']    
#     kmean = KMeans_Self(k)
#     labels = kmean.fit(data)
#     for label,cood in zip(labels,data):
#         plt.scatter(cood[0],cood[1],c=color[int(label)])
#     plt.title("kmeans by self \n K = {}".format(k))

if __name__ == "__main__":
    k = 3
    bw = 2.0
    
    plt.figure("Group 20: GT result")
    # plt.subplot(131)
    data = create_dataset().T
    # plt.subplot(132)
    # kmean_self(data,k)


    plt.figure("Group 20: KMeans result")
    plt.subplot(131)
    kmean_forbidden(data,k)
    plt.subplot(132)
    kmean_forbidden(data,6)    
    plt.subplot(133)
    kmean_forbidden(data,9)

    plt.figure("Group 20: MeanShift result")
    plt.subplot(131)
    meanShift_forbidden(data,bw)    
    plt.subplot(132)
    meanShift_forbidden(data,1.5)    
    plt.subplot(133)
    meanShift_forbidden(data,1.0)

    
    
    plt.show()


