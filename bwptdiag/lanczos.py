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


# PERFORMS THE STANDARD LANCZOS ITERATIVE DIAGONALIZATION TECHNIQUE
# BY: ZACHARY W. WINDOM, QUANTUM THEORY PROJECT, UNIVERISTY OF FLORIDA
#
# Performs the Lanczos algorithm; computes A=V^*TV if the dimensions
# are correct. The Lanczos algo transforms the eigendecomposition problem for A
# into the eigendecomposition problem for T.
#
# Input: Hermitian matrix, 'A'.
#        Dimension of A (m x m), 'dim'
#        Dimension of initial vector qj, 'itr'. 'itr'==n and qj is an element of C^n
#        Maximum number of iterations, 'maxitr'
#        Initial guess vector, 'p'
#        Convergence tolerance, 'TOL'
#        Control of printing verbosity, 'PRINT', from low (0) to high (2)
# Output: List of residuals, eigenvectors, and eigenvalues
#
#  SIZE(A)=m x m, SIZE(V)=n x m, and SIZE(T)=m x m
# ** NOTE ** If n=m, then V is unitary and A=V^*TV

def buildTk(alpha,beta,itr):
    Tk=np.diag(alpha)
    for i in range(1,itr):
        Tk[i-1,i]=beta[i]
        Tk[i,i-1]=Tk[i-1,i]
    return Tk

 
def Get_LanczosResids(HR,R,roots,vecs,i,vecsLength):
    residList=[]
    for jj in range(vecsLength):
        residEqns=(HR[:,:i]-roots[jj]*R[:,1:i+1])@vecs[:,jj]
        residEqns=np.linalg.norm(residEqns)/np.linalg.norm(R[:,1:i+1]@vecs[:,jj])
        residList.append(residEqns)
    return residList



# Program: Designed to perform the standard Lanczos iteration to iteratively diagonalize a matrix
#          Can also serve as an intermediary with the Precondition Lanczos algorithm, provided an
#          initial guess is supplied. 
# By: Zachary W. Windom 6/6/2022
#
# Amat - Matrix to be diagonalized
# dim - dimension of matrix 'Amat'
# maxitr - Max. number of Lanczos iterations
# p     - Initial guess vector
# TOL   - Metric to ensure the norm of residual vector is converged
# PRINT - Manages extra printing statements to facilitate debugging
# PreLanczosBOOL - Boolean that dictates whether 'Run_Lanczos' is being 
#                     called by 'Run_PrecondLanczos' (True), or not (False). Default is False.
def Run_Lanczos(Amat,dim=3,maxitr=10,p=np.asarray([]),TOL=10**-8,PRINT=0,PreLanczosBOOL=False,spectraType='lowest'):
    print('**** Starting Lanczos iterative diagonalization ****  \n\n')
    
    # Setup debug printing statements
    def printLanczos_Initial():
        print('Initial guess supplied to algorithm? ', p.size==0)
        print('Initial guess defined inside Lanczos: ', sum(p),'\n \n')
        print('Checking matrix & vector arrays are storing information via np.sum() ..... ')
        print('sum(R[:,0]): ', sum(R[:,0]),'\n sum(R[:,1])',sum(R[:,1]))
        print('Convergence tolerance: ',TOL)
        print('Maximum number of iterations: ', maxitr)
        print('Rank of matrix: ', dim)
        assert sum(R[:,0])==0.0
        
    def printLanczos_IterationAllInfo(PRINT):
        print('***************************************************')
        print('*************************************************** \n')
        print('PRINTING information at iteration: ', i,'\n')
        print('value of PRINT: ',PRINT)
        if PRINT==2: 
            print('Tk: \n',Tk)
            print('Current list of lowest energy solutions: \n', roots[:i])

        print('Current Ritz estimate of lowest root: ',roots[0])
        print('Current residual: ',resid)
        print('Converged lowest root? ', resid<TOL,'\n \n')
        
    # Initialize subspaces, storage arrays, initial guess vector and eval  
    R=np.zeros((dim,maxitr+2))
    HR=np.zeros((dim,maxitr))
    alphas=[]
    betas=[]
    betas.append(0.0000)
    # Initialize first iteration of Lanczos
    if p.size==0:
        blockDim=1
        p,theta,lowestElements = Get_p(Amat,dim,blockDim,True)
    else:
        theta=(p.T@Amat)@p
    R[:,1]=p[:,0]
    
    # Initialize arrays to hold convergence info
    SSevalList=[]
    MatrixVecMultiplies=[]
    ResidList=[]
    SSevalList.append(np.asarray(theta))
    
    
    for i in range(1,maxitr+1):
        print('\n\n Starting iteration: ',i)
        # Construct alpha, beta, and the residual
        HR[:,i-1]=Amat@R[:,i]
        MatrixVecMultiplies.append(i)
        alpha=(R[:,i].T).dot(HR[:,i-1])
        if i==1 and PRINT==2:
            printLanczos_Initial()
        alphas.append(alpha)
        LZresid=HR[:,i-1]-alphas[i-1]*R[:,i]-betas[i-1]*R[:,i-1]
        beta=np.linalg.norm(LZresid)
        betas.append(beta)
        R[:,i+1]=LZresid/beta
        if i==1:
            ResidList.append(np.asarray(beta))
            print('Initial estimate of root: ',alpha)
            print('Initial norm: ',beta)
        
        # Construct subspace matrix, Tk, diagonalize it, extract roots/vecs, build real residual, 
        # then test for convergence of ||<r|r>||
        if i != 1:
            Tk=buildTk(alphas,betas,i)
            roots,vecs=eigh(Tk)
            if spectraType=='lowest':
                indx=roots.argsort()
            elif spectraType=='highest':
                indx=roots.argsort()[::-1]

            roots=roots[indx]
            SSevalList.append(roots)
            vecs=vecs[:,indx]
            
            newVec=R[:,1:i+1]@vecs[:,0]
            nextTargetVec=R[:,1:i+1]@vecs[:,1]
            convergedEval=((newVec.T@Amat)@newVec)
            currentResids=Get_LanczosResids(HR,R,roots,vecs,i,len(vecs))
            resid=currentResids[0]
            ResidList.append(np.asarray(currentResids))
            print("Estimate of root: ",roots[0])
            print('Residual norm: ',resid)
            if PRINT==2:
                printLanczos_IterationAllInfo(PRINT)
            
            # If both preconditioned Lanczos is calling this program *AND* the stopping
            # condition laid out in Scott et al. 10.1137/0914037 is satisfied, then return 
            # the eigenvector back to Preconditioned Lanczos.
            if PreLanczosBOOL:
                print('Debugging stopping criteria: ')
                print('- roots[0]',-1.0*roots[0])
                print('residual: ', resid)
            if PreLanczosBOOL and -1.0*roots[0]-resid>TOL:
                print("Success! Standard Lanczos has reached the Preconditioned Lanczos stopping criteria in ",i," iterations.")
                return i, newVec,ResidList,SSevalList,nextTargetVec
                
            # Determine if the standard Lanczos iteration has converged the lowest soln.
            if resid < TOL:
                print('\n ** LANCZOS CONVERGED LOWEST ROOT ON ITERATION: ',i)
                print('Value of lowest root: ',roots[0])
                print('Residual information: ',resid,'\n\n')
                break
    

    # returns the iteration # the lowest root converges, the converged eigenvector, 
    # list of residuals, list of subspace eigenvalues
    return i, newVec,ResidList,SSevalList,nextTargetVec #convergedEigVector
    


