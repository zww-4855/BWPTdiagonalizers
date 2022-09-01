import numpy as np
from numpy import linalg as LA
from scipy import linalg
from scipy.linalg import eigh
#import matplotlib.pyplot as plt
import math
import sys as sys
import pprint, pickle

###
### Files to ease the handling of IO operations
### TO DO::
### ADD: fxn to write residuals to output file, maybe plot as well

def createModelMatrix(matrixDim,sparsity):
    #matrixDim=10
    #sparsity = 0.01
    Amat = np.zeros((matrixDim,matrixDim))
    for i in range(0,matrixDim):
        Amat[i,i] = i + 1
    Amat = Amat + sparsity*np.random.randint(20,size=(matrixDim,matrixDim))
    #print("Amat: \n",Amat)
    Amat=(Amat + Amat.T)/2
    return Amat

def loadMatrixFile(filename,dataType=''):
    if dataType=='':
        print('no datatype specified!!!')
        Amat=np.loadtxt(filename,dtype = np.float64)#ulonglong)
        dim=int((Amat.size)**(1.0/2.0))
        Amat=np.reshape(Amat,(dim,dim),order='F')
        return Amat,dim
    elif dataType=='binary':
        Amat=np.fromfile(filename, dtype=np.float64)
        dim=int((Amat.size)**(1.0/2.0))
        Amat=np.reshape(Amat,(dim,dim),order='F')
        return Amat,dim



# Dump convergence information to file, AND call plotting software
# Call at the conclusion of iterative diagonalization 
def postProcessInfo(filename, matVecMult,theta, resid):
    root1Info,root2Info,root3Info=zip(matVecMult,theta, resid)
    sortedInfo=[root1Info,root2Info,root3Info]
    saveResidInfo(filename,sortedInfo)
    return sortedInfo
    
# Call at the conclusion of Davidson **AND**
# when multiple diagonalization strategies have been employed and their results dumped to file
# diagStratLabels=['Stnd Davidson', 'BW-PT DAvidson',etc]
# INPUT: 4D array 'sortedInfoAll' -- the 3 indices explained above "def loadResidInfo", seen below
#     pltTitle          -- Title of the .eps image plotting convergence behavior
#     diagStratLabels      -- array of Legend labels, denoting the various diagonalization strategies studied
#     pltResid          -- Boolean variable the dictates whether the y-axis is residual or diff. b/t exact
#                    and approximated eigenvalues
#     convrgRootList          -- array that stores the exact, 3 lowest eigenvalues of the matrix in question 
#        dimension 0 on the interval (1,3*number of methods): refers to root 0,1,2 for a part. method
#        dimension 1 on the interval (0,2) : 0 is iteration count, 1 is est. root value, 2 is residual
# **CURRENTLY CAN PLOT MULTPLY INPUTS, BUT ONLY FOR RESIDUAL AND NOT EST. ROOT VALUES ZWW 3/8/22
#def pltConvergenceInfo(sortedInfoAll,pltTitle,diagStratLabels=None,pltResid=True,convrgRootList=[0.0000,0.0000,0.0000]):
#    fig, (ax1,ax2,ax3)=plt.subplots(1,3,figsize=(16,5))
#    colors=['k','red','limegreen','mediumblue','gold','c']
#    
#    markerLabels=['<','s','P','*','D',6,'o']
#    if pltResid: # residuals are the 2 index
#        indx=2
#    else:
#        indx=1
#
#    for xx in range(len(sortedInfoAll)):
#        sortedInfo=sortedInfoAll[xx]
#        counter=0
#        for i in range(len(sortedInfo)):
#            Eeval=np.repeat(convrgRootList[i%3],len(sortedInfo[i][indx]))
#            if i%3==0:
#                ax1.plot(sortedInfo[i][0],sortedInfo[i][indx]-Eeval,colors[counter],marker=markerLabels[counter])
#            if i%3==1:
#                ax2.plot(sortedInfo[i][0],sortedInfo[i][indx]-Eeval,colors[counter],marker=markerLabels[counter])
#
#            if i%3==2:
#                ax3.plot(sortedInfo[i][0],sortedInfo[i][indx]-Eeval,colors[counter],marker=markerLabels[counter],label=diagStratLabels[counter])
#                counter=counter+1
#
#    ax1.set_yscale('log')
#    ax2.set_yscale('log')
#    ax3.set_yscale('log')
#    ax3.legend(prop={"size":14})
#    ax1.set_title('Root 1',fontsize=18)
#    if pltResid:
#        ax1.set_ylabel('Residual',fontsize=16)
#    else:
#        ax1.set_ylabel(r'$\lambda - \lambda_i$',fontsize=16)
#    ax1.set_xlabel('Matrix-Vector Multiplies',fontsize=16)
#    ax2.set_xlabel('Matrix-Vector Multiplies',fontsize=16)
#    ax3.set_xlabel('Matrix-Vector Multiplies',fontsize=16)
#    ax2.set_title('Root 2',fontsize=18)
#    ax3.set_title('Root 3',fontsize=18)
#    plt.savefig('%s.eps'%pltTitle,format='eps',dpi=1000)
#    plt.show()
    
    

# Example resid info:
# Ex. for one diag strat/mult. roots:
#allD=[]
#one=[[1,1.0],[2,0.69],[3,0.00001]]
#two=[[1,0.1],[2,0.0000000007],[3,0.0]]
#allD.append(one)
#allD.append(two)
#print(allD,allD[1])

# Dumps residual info for multiple roots at a time to a file; index 0 pertains to a root, index 1 is convergence info for that root -- ie iteration #, estimated theta, and residual -- and index 2 is cycles between the 3 elements in the array
# Ex: residInfo=[[[1,0.6,0.33333],[2,0.0001,0.303]],[[1,0.89,0.111],[2,0.801,10^-6]]]
def saveResidInfo(filename,residInfo3D):
    output = open(filename, 'wb')
    pickle.dump(residInfo3D, output)
    output.close()


# input: filename_s = ['filea','fileb',etc]
# output: residInfo3D if filename has only one file
# output: residInfo4D otherwise
# Relevant when plotting residInfo for multiple roots, for multiple Diagonalization strategies
# For multiple stored files: index 0 is *all* information corresponding to the filename_s ordering
#                 index 1 returns filename_s' matrix-vec. multiply,eigval., and residual information for a particular root
#                 index 2 returns either matrix-vec. multiply, eig. value, or residual information

def loadResidInfo(filename_s):
    residInfo4D=[]
    for file in filename_s:
        pkl_file = open(file, 'rb')
        residInfo3D = pickle.load(pkl_file)
        pprint.pprint(residInfo3D)
        pkl_file.close()
        if (len(filename_s) > 1):
            residInfo4D.append(residInfo3D)
       
    return residInfo3D,residInfo4D


