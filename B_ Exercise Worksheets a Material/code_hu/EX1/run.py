import numpy as np
import matplotlib.pyplot as plt
from PA_Exercise_02 import Exercise_1
from PA_Exercise_02 import Exercise_2
from PA_Exercise_02 import CrossValidation as CV

SHOW_ASSIGNMENT = False
SHOW_OBSERVE_1 = False
SHOW_OBSERVE_2 = False
SHOW_CV = True

if __name__=="__main__":
    ex1 = Exercise_1(3)
    ex2 = Exercise_2()

    samples = [10000,50000,100000,200000,400000] 
    weights = [9,40,60]

    if SHOW_ASSIGNMENT:
        img_1 = ex1.get_img()
        img_2 = ex1.do_sampling(100000)
        img_3 = ex2.do_sampling(img_2,9)
        
        plt.figure("Group 20: Exercise 02 result", figsize=(10,3))
        plt.subplot(131)
        plt.imshow(img_1,cmap="gray")
        plt.title("Gaussian filter with \\sigema = {}".format(3))
        plt.subplot(132)
        plt.imshow(img_2)
        plt.title("New img with sample = {}".format(100000))        
        plt.subplot(133)
        plt.imshow(img_3)
        plt.title("New img with kernel = {}".format(9))
        plt.suptitle("Group 20: Exercise 02 result")

    if SHOW_OBSERVE_1:
        idx = 1
        plt.figure("Observe from exercise 1")
        sampleNum = len(samples)
        for sample in samples:
            img = ex1.do_sampling(sample)            
            plt.subplot(1,sampleNum,idx)
            plt.imshow(img)
            plt.title("New img with sample = {}".format(sample)) 
            idx += 1
        plt.suptitle("Observe from exercise 1") 
    
    if SHOW_OBSERVE_2:
        idx = 1
        plt.figure("Observe from exercise 2")
        weightNum = len(weights)+1
        imgInput = ex1.do_sampling(100000)
        plt.subplot(1,weightNum,idx)
        plt.imshow(imgInput)
        plt.title("Input img with sample = {}".format(100000)) 
            
        for weight in weights:
            idx += 1
            img = ex2.do_sampling(imgInput, weight)            
            plt.subplot(1,weightNum,idx)
            plt.imshow(img)
            plt.title("New img with K-weight = {}".format(weight)) 
        plt.suptitle("Observe from exercise 2") 
    
    if SHOW_CV:
        sampleSize = [100000]
        k = 10

        candidateMin = 7
        candidateMax = 15        
        candidate = np.linspace(candidateMin, candidateMax, candidateMax-candidateMin+1)

        gtd = ex1.get_img()
        cv = CV(k, candidate, gtd, (ex1,ex2))
        
        for sample in sampleSize:                
            res = cv.train(ex1.get_idx(sample))
            cv.plotResult(res,sample)

    plt.show()

