import numpy as np
from numpy import linalg as LA
from scipy import linalg
from scipy.linalg import eigh
import math
import sys as sys
import os as os

def computeRHR(R,HR,i,blockDim):
#    print('size of RHR should be ',i+1,i+1)
    RHR=R[:,:blockDim+i].T@HR[:,:blockDim+i]
#    print('actual size of RHR:',np.shape(RHR))
    print('RHR: \n',RHR)
#    print('transpose RHR: \n', HR[:,:blockDim+i].T@R[:,:blockDim+i])
#    print('print mini RHR: \n',R[:,:blockDim+i-1].T@HR[:,:blockDim+i-1])
#    print('dot prod: ', np.dot(R[:,0],HR[:,1]),np.dot(R[:,1],HR[:,0]))
    print('Debug: Make sure RHR==RHR.T ie is Hermitian',np.allclose(RHR,RHR.T,rtol=10**-15,atol=10**-15))
    roots,vecs=np.linalg.eig(RHR)
#    print('Roots from <R|H|R> via QR:',np.sort(roots))
    indx=roots.argsort()
    theta=roots[indx]
    vecs=vecs[:,indx]
    print('Returning ......')
    print('Sorted Ritz estimates: ',theta[:5])
#    print('C_i vecs: ',vecs)

    #print('C_i vecs: ',vecs[:,usableIndx])
    return theta,vecs


def GramSchmidt(A,nrows,ncols):
    for j in range(ncols):
        for k in range(j):
            temp=np.dot(A[:,k],A[:,j])
            A[:,j]=A[:,j]-temp*A[:,k]
        A[:,j]=A[:,j]/np.linalg.norm(A[:,j])

    return A

def Get_CorrVec(R,residual,theta,H0,matrixDim,i,target=0,H0def='diag'):
    if H0def=='diag' or H0def=='PHP':
        R[:,[i]]=SimpleT0(residual,theta,H0,matrixDim,blockDim,target,H0def)
    else:
        print('Do not have this H0def yet built into Get_CorrVec')
        sys.exit()
    
    return R
        
def SimpleT0(residual,theta,H0,matrixDim,blockDim,target=0,H0def='diag'):
    corrVec=np.zeros((matrixDim,blockDim))
    QH0Q=np.zeros((matrixDim,matrixDim))
    print('shapes: ',np.shape(corrVec),np.shape(residual))
    if H0def=='diag':
        QH0Q=H0
    elif H0def=='PHP':
        print('continue')
        
    for zzz in range(matrixDim):
        xxx=theta-QH0Q[zzz][zzz]
        if (abs(xxx) < 0.0001):
            xxx=math.copysign(0.0001, xxx)
        corrVec[zzz,[0]]=residual[zzz]/xxx
    
    return corrVec

def Get_residual(H,R,HR,vecs,i,matrixDim,blockDim,p,theta,target=0):
    residual=np.zeros((matrixDim,blockDim))
    print('@Get_resid: value of i: ',i)
    if i==0:
        
        residual[:,0]=HR[:,0]-theta*R[:,0]
    else:
        #print(np.shape(HR[:,:i]),np.shape(vecs[:,target]),np.shape(R[:,:i]),np.shape(theta[0]))
        phi=R[:,:i]@vecs[:,[target]]
        print('test eigenvalue: ',(phi.T@H)@phi)
        residual[:]=HR[:,:i]@vecs[:,[target]]-theta*(R[:,:i]@vecs[:,[target]])
        
    normResid=np.linalg.norm(residual)
    print('norm resid',normResid)
    #print('residual ',residual)
    return residual,normResid





