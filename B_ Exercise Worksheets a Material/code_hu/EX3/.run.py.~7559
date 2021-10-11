import numpy as np

import scipy.misc
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection  import cross_val_score
import matplotlib.pyplot as plt
 
DeBug = 0
class Racoon:
    def __init__(self,img,sampleNum):
        self.x = self.get_x(img)
        self.y = self.get_y(img,sampleNum)

    def get_data(self):
        return self.x,self.y

    def get_x(self,img):
        m,n = img.shape
        v = np.linspace(0,m-1,m)
        h = np.linspace(0,n-1,n)
        xv,yv = np.meshgrid(h,v)
        x = np.zeros([m*n,2])
        x[:,0] = xv.flatten()
        x[:,1] = yv.flatten()
        return x

    def get_y(self,img,sampleNum):
        cvf = np.cumsum(img)
        cdf = cvf/np.max(cvf)        
        randomP = np.random.uniform(0,1.,sampleNum)
        idx = np.searchsorted(cdf, randomP)        
        imgNew = np.zeros_like(img)      

        idx_fore = idx[0:int(sampleNum/2)]    
        idx_back = idx[int(sampleNum/2):,]
        
        x = idx_fore // img.shape[1]
        y = idx_fore % img.shape[1]
        imgNew[x,y] = 1 #img[x,y]
        
        x = idx_back // img.shape[1]
        y = idx_back % img.shape[1]
        imgNew[x,y] = np.random.uniform(0,1.,1)
        if DeBug:
            plt.figure("Image with sample = {}".format(sampleNum))
            plt.imshow(imgNew)   
        return imgNew.flatten()

class CrossValidation:
    def __init__(self, k,data,scoring):        
        self.rf = RandomForestRegressor
        self.ef = ExtraTreesRegressor
        self.k = k
        self.x = data[0]
        self.y = data[1]
        self.scoring = scoring

    def run(self, method, candidates):
        res = []
        if method=="randomForest":
            forest = self.rf
        else:
            forest = self.ef

        for candidate in candidates:
            regressor = forest(n_estimators=int(candidate[0]),max_depth=int(candidate[1]))
            scores = cross_val_score(regressor,self.x,self.y,cv=self.k,scoring=self.scoring)
            res.append(np.sum(scores)/self.k)
        return np.array(res)

if __name__ == "__main__":
    img = scipy.misc.face(gray=True) # 1024 * 786 = 700 0000 
    SampleNum = 1000000
    racoon = Racoon(img,SampleNum) # 100 0000
    data = racoon.get_data() # flatten
    if DeBug:
        print(x.shape,y.shape)
    
    minN = 18
    maxN = 27

    minD = 5
    maxD = 14

    # candidate = np.ones([10,2]) * 10
    # n = np.linspace(minN,maxN,10)
    # candidate[:,0] = n
    
    # plt.figure("Group 20: Random Forest Cross Validation Result")
    # plt.subplot(121)
    # cv = CrossValidation(3,data,"neg_mean_squared_error")
    # res = cv.run("randomForest", candidate)    
    # idxN = np.where(res==np.max(res))
    # optN = int(n[idxN])
    # plt.title("CV result with tree number = {} - {} \n opt = {}".format(minN,maxN,optN))
    # plt.plot(n,res,"-b")
    # plt.plot(n[idxN],res[idxN],"*r")
    # plt.xlabel("number of tree")
    # plt.ylabel("Negative Mean Squared Error")
    # plt.legend(["theta","optimal"])


    # plt.subplot(122)
    # candidate = np.ones([10,2]) * optN
    # d = np.linspace(minD,maxD,10)
    # candidate[:,1] = d
    # res = cv.run("randomForest", candidate)   
    
    # idxD = np.where(res==np.max(res))
    # optD = int(d[idxD]) 
    # plt.title("CV result with maximum depth = {} - {} \n opt = {}".format(minD,maxD,optD))

    # plt.plot(d,res,"-b")
    # plt.plot(d[idxD],res[idxD],"*r")
    # plt.xlabel("Maximun Depth")
    # plt.ylabel("Negative Mean Squared Error")
    # plt.legend(["theta","optimal"])

    # plt.suptitle("CV with samples ={}".format(SampleNum))
    optN = 13
    optD = 13

    rf = RandomForestRegressor(n_estimators=int(optN),max_depth=int(optD))
    ef = ExtraTreesRegressor(n_estimators=int(optN),max_depth=int(optD))
    y_rf = rf.fit(data[0],data[1]).predict(data[0])
    y_ef = ef.fit(data[0],data[1]).predict(data[0])

    plt.figure("Group 20: RFR and ETR Image With Optimal value")
    plt.subplot(121)
    plt.title("Random Forest")
    plt.imshow(y_rf.reshape([768,1024])) 
    plt.subplot(122)    
    plt.title("ExtraTree Forest")
    plt.imshow(y_ef.reshape([768,1024])) 
    
    plt.suptitle("RF & EF predict image with optimal parameter numTree = {}, maxDepth = {}".format(optN,optD))
    if DeBug:
        print(y_rf)
    plt.show()