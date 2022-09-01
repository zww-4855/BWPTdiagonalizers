import numpy as np
from numpy import linalg as LA
from scipy import linalg
from scipy.linalg import eigh
import math
import sys as sys
import os as os

# Returns initial guess, p, and estimate of ritz root, theta
def Get_p(Amat,matrixDim,blockDim,DEBUG=True,iterH0def=""):
    p=np.zeros((matrixDim,blockDim))
    
    diagMat=np.diag(Amat)
    global lowestElements
    lowestElements=np.argsort(diagMat)
    if DEBUG:
        print('Using column of identity matrix as initial guess?',iterH0def=="")
        print('SHAPE OF p', np.shape(p))
        print('diagonal sorted: ',np.sort(diagMat))
        print('lowestElements: ',lowestElements)
        

    if iterH0def=="":
        print('loading column of identity as p')
        for i in range(blockDim):
            p[lowestElements[i],i]=1.0
    else:
        for i in range(2):
            p[lowestElements[i],0]=1.0
        p[:,0]=p[:,0]/np.linalg.norm(p[:,0])
    
    guessEps=[]
    for x in range(blockDim):
        guessEps.append(diagMat[lowestElements[x]])# -0.001) # seed starting E's to be smaller than diagonal elements


    theta=guessEps[0]
    print('Initial BW-PT guesses to eigenvalues:',theta)
    return p,theta,lowestElements

def Get_H0(Amat,p,matrixDim,blockDim,H0def,blockSize=0):
    projP=np.outer(p,p)
    projQ=np.identity(matrixDim) - projP
    H0=np.zeros((matrixDim,matrixDim))
    if H0def=='PHP':
        H0=(projP@Amat)@projP
        V=Amat-H0
        #print('V \n',V)
        #print('original Guess:',diagMat[0])
        H=Amat

    elif H0def=='diag' or H0def=='dynamicH0Diag':
        H0=np.diag(np.diag(Amat))
        V=Amat-H0


    elif H0def=='10diag':
        H0=np.diag(np.diag(Amat))
        for xx in range(10):
            offDiag=np.diag(Amat,k=xx+1)
            for yy in range(len(offDiag)):
                H0[yy+xx+1,yy]=offDiag[yy]
                H0[yy,yy+xx+1]=offDiag[yy]

        return H0,Amat-H0




    elif H0def=='QuadDiag' or H0def=='TriDiag' or H0def=='BiDiag':
        H0=np.diag(np.diag(Amat))

        offDiag=np.diag(Amat,k=1)
        offDiag2=np.diag(Amat,k=2)
        for zzz in range(len(offDiag)):
            H0[zzz+1,zzz]=abs(offDiag[zzz])
            H0[zzz,zzz+1]=abs(offDiag[zzz])


        if H0def=='BiDiag': return H0,Amat-H0

        for yyy in range(len(offDiag2)):
            H0[yyy+2,yyy]=offDiag2[yyy]
            H0[yyy,yyy+2]=offDiag2[yyy]
      
        if H0def=='TriDiag': return H0,Amat-H0

        for yyy in range(len(np.diag(Amat,k=3))):
            H0[yyy+3,yyy]=np.diag(Amat,k=3)[yyy]
            H0[yyy,yyy+3]=np.diag(Amat,k=3)[yyy]

    
        V=Amat-H0

    elif H0def=='block':
        print('Have not added block logic yet')
        sys.exit()
        
    else:
        print('Not a viable partitioning choice; please choose from / PHP / or / diag /')
        sys.exit()

    return H0,V



def Get_InitialSpace(Amat,p,theta,Hdim,blockDim,maxItr):
    HR=np.zeros((Hdim,blockDim*maxItr+1))  # stores all matrix vector multiplies of H w/ the vectors in R
    R=np.zeros((Hdim,blockDim*maxItr+1)) # stores {|p>, and all correction vecs |Phi>}
    R[:,np.arange(0,blockDim)]=p
    HR[:,np.arange(0,blockDim)]=Amat@p
    return R, HR






