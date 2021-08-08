import numpy as np
from sklearn.neighbors import KDTree

def LPM_cosF(neighborX, neighborY, lbd, vec, d2, tau, K):
    L = neighborX.shape[0]
    C = 0
    Km = np.array([K+2, K, K-2])
    M = len(Km)

    for KK in Km:
        neighborX = neighborX[:,1:KK+1]
        neighborY = neighborY[:,1:KK+1]

        ## This is a loop implementation for computing c1 and c2, much slower but more readable
        # ni = np.zeros((L,1))
        # c1 = np.zeros((L,1))
        # c2 = np.zeros((L,1))
        # for i in range(L):
        #     inters = np.intersect1d(neighborX[i,:], neighborY[i,:])
        #     ni[i] = len(inters)
        #     c1[i] = KK - ni[i]
        #     cos_sita = np.sum(vec[inters, :]*vec[i,:],axis=1)/np.sqrt(d2[inters]*d2[i]).reshape(ni[i].astype('int').item(), 1)
        #     ratio = np.minimum(d2[inters], d2[i])/np.maximum(d2[inters], d2[i])
        #     ratio = ratio.reshape(-1,1)
        #     label = cos_sita*ratio < tau
        #     c2[i] = np.sum(label.astype('float64'))

        neighborIndex = np.hstack((neighborX,neighborY))
        index = np.sort(neighborIndex,axis=1)
        temp1 = np.hstack((np.diff(index,axis = 1),np.ones((L,1))))
        temp2 = (temp1==0).astype('int')
        ni = np.sum(temp2,axis=1)
        c1 = KK - ni
        temp3 = np.tile(vec.reshape((vec.shape[0],1,vec.shape[1])),(1,index.shape[1],1))*vec[index, :]
        temp4 = np.tile(d2.reshape((d2.shape[0],1)),(1,index.shape[1]))
        temp5 = d2[index]*temp4
        cos_sita = np.sum(temp3,axis=2).reshape((temp3.shape[0],temp3.shape[1]))/np.sqrt(temp5)
        ratio = np.minimum(d2[index], temp4)/np.maximum(d2[index], temp4)
        label = cos_sita*ratio < tau
        label = label.astype('int')
        c2 = np.sum(label*temp2,axis=1)

        C = C + (c1 + c2)/KK


    idx = np.where((C/M) <= lbd)
    return idx[0], C

def LPM_filter(X, Y):
    lambda1 = 0.8   
    lambda2 = 0.5  
    numNeigh1 = 6    
    numNeigh2 = 6
    tau1 = 0.2  
    tau2 = 0.2

    vec = Y - X
    d2 = np.sum(vec**2,axis=1)

    treeX = KDTree(X)
    _, neighborX = treeX.query(X, k=numNeigh1+3)
    treeY = KDTree(Y)
    _, neighborY = treeY.query(Y, k=numNeigh1+3)

    idx, C = LPM_cosF(neighborX, neighborY, lambda1, vec, d2, tau1, numNeigh1)

    if len(idx) >= numNeigh2 + 4:
        treeX2 = KDTree(X[idx,:])
        _, neighborX2 = treeX2.query(X, k=numNeigh2+3)
        treeY2 = KDTree(Y[idx,:])
        _, neighborY2 = treeY2.query(Y, k=numNeigh2+3)
        neighborX2 = idx[neighborX2]
        neighborY2 = idx[neighborY2]
        idx, C = LPM_cosF(neighborX2, neighborY2, lambda2, vec, d2, tau2, numNeigh2)

    mask = np.zeros((X.shape[0],1))
    mask[idx] = 1

    return mask.flatten().astype('bool')



