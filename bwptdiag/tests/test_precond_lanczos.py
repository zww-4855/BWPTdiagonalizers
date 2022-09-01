"""
Unit and regression test for the bwptdiag package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

from bwptdiag.preconditioned_lanczos import Run_PrecondLanczos

import numpy as np

# # EXAMPLE 1 OF SCOTTS PRECONDITIONED LANCZOS PAPER: NORMAL LANCZOS


def test_lanczos():
    """Sample test, will always pass so long as import statement worked."""
    assert "bwptdiag" in sys.modules
    A=np.diag(np.arange(1,1001))
#print(A)
    M=np.arange(10.1,110.1,0.1)
    initp=np.repeat(1,1000)
    #print(initp)
    p=np.zeros((1000,))
    for i in range(1,1001):
        #print(initp[i-1],float(i),initp[i-1]/float(i))
        p[i-1]=initp[i-1]/float(i)
    
    E=(p.T@A)@p/(p.T@p)
    print('E',E)
    Mk=np.diag(M)-np.diag(np.repeat(E,1000))
    print(M)
    Lk=np.linalg.cholesky(Mk)
    print(Lk)
    p=p/np.linalg.norm(p)
    p=p.reshape((1000,))
    newp=p
    print(Mk)
    matrixVecMultiply, ResidList,SSevalList=Run_PrecondLanczos(A,1000,H0def="diag",microItr=50,macroItr=5,
                                                               TOL=10**-8,p=newp,
                                                               QHQapprox="diag",PreLanczosBOOL=True,customMk=Mk)
