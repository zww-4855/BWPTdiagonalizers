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

def Run_RedefineH(Amat,dim,maxitr,p=np.asarray([]),TOL=10**-8)
    R = np.zeros((dim, maxitr + 2))
    HR = np.zeros((dim, maxitr))
    if p.size == 0:
        blockDim = 1
        p, theta, lowestElements = Get_p(Amat, dim, blockDim, True)


    #Construct P
    for i in range(maxitr):
        projP=np.outer(p,p)
        projQ=np.eye(dim)-projP
        print(np.shape(projP))
        #construct approx H
        Hhat=(projP@Amat)@projP + (projP@Amat)@projQ +(projQ@Amat)@projP + (projQ@Amat)@projQ


        # Solve Hhat exactly with Davidson
