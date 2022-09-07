import numpy as np
from numpy import linalg as LA
import scipy.linalg
from scipy import linalg
from scipy.linalg import eigh
import math
import sys as sys
import os as os
from bwptdiag.InitializeBWPT import *
from bwptdiag.NumericalRoutines import *
from bwptdiag.IOhandling import *
from bwptdiag.davidson import *

def Run_RedefineH(Amat,dim,maxitr,p=np.asarray([]),TOL=10**-8):
    R = np.zeros((dim, maxitr + 2))
    HR = np.zeros((dim, maxitr))
    if p.size == 0:
        blockDim = 1
        p, theta, lowestElements = Get_p(Amat, dim, blockDim, True)

    projP=np.zeros((dim,dim))
    projQ=np.eye(dim)
    #Construct P
    for i in range(maxitr):
        projP=np.outer(p,p)
        projQ=np.eye(dim)-projP#+=-1.0*projP
        np.allclose(projP@projP, projP)
        np.allclose(projQ@projQ,projQ)
        print(np.shape(projP))
        #construct approx H
        approxH=Amat#np.diag(np.diag(Amat))
        QHhatQ=(projQ@approxH)@projQ#np.diag(np.diag((projQ@approxH)@projQ))
        Hhat=(projP@Amat)@projP + (projP@Amat)@projQ +(projQ@Amat)@projP + QHhatQ#(projQ@approxH)@projQ

        u,v=np.linalg.eig(Hhat)
        print('Sorted roots of Hhat:', np.sort(u)[:6])
        # Solve Hhat exactly with Davidson
        matmul,thetas,resids,SSevals,eVec=Run_DavidsonBWPT(Hhat, dim, 1 ,'temp.txt', p,theta,TOL=10**-10,\
                     maxItr=10,solnCount=1,H0def='diag',highOresolvent=OrderedDict(),invBOOL="QHQ",\
                    numSSVecs_keep=15,maxSSvecs=50,iterH0def="diag",spectrum='lowest')


        #normalize eVec, set equal to p
        p=eVec/np.linalg.norm(eVec)
        theta=(p.T @Hhat)@p
        print('test of <p|Hhat|p>: ', (p.T @Hhat)@p)
        print('resids: ', resids)
        print('matmuls:',matmul)
        p=np.reshape(p,(dim,1))
        print('shape of p: ', np.shape(p))
        np.reshape(p,(dim,1))

