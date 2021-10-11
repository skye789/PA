import numpy as np
from numpy.linalg import inv
class KernelDensity:
    def __init__(self,x,y,bw=0.2):

        self.X = x
        self.Y = y
        self.bw=bw
        N = np.size(x)
        B = np.ones((N,2))
        B[:,1] = x
        self.B = B 
        self.N = N

    def dFunction(self,t):
        # t[t<-1] = 0
        # t[t>1] = 0
        # t = 3/4 * (1 - t**2)
        # t[t==0.75] = 0

        for i in range(np.size(t)):
            if np.abs(t[i]) <=1:
                t[i] = 3/4 * (1 - t[i]**2)
            else :
                t[i] = 0
        return t

    def kFunction(self,x0,bw):
        xi = self.X
        t = (xi-x0)/bw
        return self.dFunction(t)

    def lFunction(self,bi,B,W,x0):
        a = bi.T
        b = np.dot(np.dot(B.T,W),B)
        c = np.dot(B.T,W)
        return np.dot(a*inv(b),c)

    def epanechnikov(self,x0):
        bw = self.bw
        yi = self.Y        
        k = self.kFunction(x0,bw)
        self.k = k
        return np.sum(k*yi)/np.sum(k)
        
    def localLinearRegression(self,x0):        
        bw = self.bw
        k = self.kFunction(x0,bw)
        w = np.zeros((self.N,self.N))
        for i in range(self.N):
            w[i,i] = k[i]
        bi =self.B[np.where(self.X==x0)[0]]  

        l = self.lFunction(bi,self.B,w,x0) 
        self.k = np.sum(l,axis=0)
        return np.sum(l*self.Y)

    def getKernel(self):
        return self.k