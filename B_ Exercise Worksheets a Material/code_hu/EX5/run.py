import numpy as np
from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans


import matplotlib.pyplot as plt

def create_dataset():
    cood0 = np.random.normal(size=(2,100))
    cood1 = np.random.normal(size=(2,100))
    cood2 = np.random.normal(size=(2,100))
    cood3 = np.random.normal(scale=0.5, size=(2,15))

    center0 = (5.0,7.5)
    center1 = (2.5,3.5)
    center2 = (8.0,4.0)
    center3 = (6.5,6.0)

    
    cood0[0] = cood0[0] + center0[0]
    cood0[1] = cood0[1] + center0[1]
    cood1[0] = cood1[0] + center1[0]
    cood1[1] = cood1[1] + center1[1]
    cood2[0] = cood2[0] + center2[0]
    cood2[1] = cood2[1] + center2[1]
    cood3[0] = cood3[0] + center3[0]
    cood3[1] = cood3[1] + center3[1]

    plt.scatter(cood0[0],cood0[1],c='r')
    plt.scatter(cood1[0],cood1[1],c='g')
    plt.scatter(cood2[0],cood2[1],c='b')
    plt.scatter(cood3[0],cood3[1],c='k')
    plt.title("ground trough")

    l0 = np.ones(cood0.shape[1]) * 0
    l1 = np.ones(cood1.shape[1]) * 1
    l2 = np.ones(cood2.shape[1]) * 2
    l3 = np.ones(cood3.shape[1]) * 3
    
    label = np.concatenate((l0, l1),axis=0)
    label = np.concatenate((label, l1),axis=0)
    label = np.concatenate((label, l1),axis=0)

    result = np.concatenate((cood0,cood1),axis=1)
    result = np.concatenate((result,cood2),axis=1)
    result = np.concatenate((result,cood3),axis=1)
    return result.T,label


def kmean_forbidden(data,k,datatype):
    color = ['r','g','b','y','c','m','k','gold','cyan','purple','magenta']
    res = KMeans(n_clusters=k).fit(data)
    labels = res.labels_
    for label,cood in zip(labels,data):
        plt.scatter(cood[0],cood[1],c=color[int(label)])
    plt.title("kmeans with {} \n K = {}".format(datatype,k))
    return labels,res.inertia_

def squared_Euclidean_distance(kLabel,center,data):
    d = 0
    for i in range(len(center)):
        cache = (data[kLabel==i][:,0] -center[i][0])**2 + (data[kLabel==i][:,1] -center[i][1])**2
        d += np.sum(cache)/(2*cache.shape[0])
    return d

def gap(wk,en):
    return np.log(en)-np.log(wk)

if __name__ == "__main__":
    plt.figure("gt")
    data,label = create_dataset()
    data = data/10
    random_data = np.random.rand(*data.shape) 
    wk_data = []
    wk_e = []
    
    plt.figure("kmean for data")
    for k in range(8):
        plt.subplot(2,4,k+1)
        kLabel,kinertia = kmean_forbidden(data,k+1,"data")
        wk_data.append(kinertia)

    plt.figure("kmean for random")
    for k in range(8):
        plt.subplot(2,4,k+1)
        rLabel,rinertia = kmean_forbidden(random_data,k+1,"random")
        wk_e.append(rinertia)

    x = np.linspace(0,7,8)
    wk_orig = wk_data - wk_data[0]
    wk_rand = wk_e - wk_e[0]
    plt.figure("wk and we")
    plt.plot(x+1,wk_orig)
    plt.plot(x+1,wk_rand)
    plt.title("wk and ewk")
    
    plt.figure("gap")
    gapk = gap(wk_data,wk_e)
    plt.plot(x+1,gapk)
    plt.title("gap")
    
    plt.show()