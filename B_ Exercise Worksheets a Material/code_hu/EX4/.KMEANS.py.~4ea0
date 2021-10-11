import numpy as np

class KMeans_Self:
    def __init__(self,K, iters=10):
        self.K = K
        self.iters = iters

    def createCenter(self,data):
        centers = []
        maxRange = np.max(data,axis=1)
        for _ in range(self.K):
            x = np.random.uniform(low=0,high=maxRange[0],size=1)
            y = np.random.uniform(low=0,high=maxRange[1],size=1)
            centers.append([int(x),int(y)])
        return np.array(centers)

    def distance(self,centers,data):
        res = np.zeros([data.shape[0],self.K])
        for i in range(self.K):
            res[:,i] = np.sqrt((centers[i,0]-data[:,0])**2+(centers[i,1]-data[:,1])**2)
        return res

    def getRnk(self,dis):  
        for i in range(dis.shape[0]):
            dis[i] = dis[i] / np.max(dis[i])

        dis[dis<1] = 0
        return dis
    
    def updateCenters(self,rnk,data):        
        centers = []
        for i in range(self.K):
            r = rnk[:,i]
            # print(r.shape,data.shape)
            x = np.sum(data[:,0] * r) / np.sum(r)
            y = np.sum(data[:,1] * r) / np.sum(r)
            centers.append([int(x),int(y)])
        return np.array(centers)

    def lossCalculate(self,rnk,centers,data):
        loss = 0
        for i in range(self.K):
            cx = rnk[:,i] * centers[i,0]
            cy = rnk[:,i] * centers[i,1]
            loss +=np.sum(np.sqrt((cx-data[:,0])**2+(cy-data[:,1])**2)) 

        return loss

    def getLabel(self,rnk):
        l = np.zeros([rnk.shape[0]])
        for i in range(rnk.shape[0]):
            l[i] = np.where(rnk[i] == 1)[0][0]
        # print(l)
        return l


    def fit(self,data):
        centers = self.createCenter(data)   
        iters = 0
        while self.iters > iters:
            iters += 1
            dis = self.distance(centers,data)
            rnk = self.getRnk(dis)
            centers = self.updateCenters(rnk,data)
            # loss = self.lossCalculate(rnk,centers,data)
            # print(loss)
            # print(centers)
        label = self.getLabel(rnk)
        return label