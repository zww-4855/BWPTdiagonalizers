import numpy as np
from numpy import linalg as LA
from scipy import linalg
from scipy.linalg import eigh
from scipy.sparse import linalg as lin
import math
import sys as sys
import os as os
from collections import OrderedDict

from bwptdiag.InitializeBWPT import *
from bwptdiag.NumericalRoutines import *
from bwptdiag.IOhandling import *

from scipy.sparse.linalg import gmres
from collections import deque

##########################################################################
# Program: Generalization of single-vector Davidson and preconditioned
# .         Lanczos using the BW-PT framework. Chooses initial guess
#          based on lowest elements of diagonal. Returns converged root(s) and the list
#          of residuals at each iteration.
# Amat - full matrix to diagonalize
# dim  - dimension of Amat
# itr  - maximum number of iterations for any root
# eigNum- number of roots user requests
# tol   - convergence criteria based on the residual
##########################################################################


print(os.getcwd())
print(sys.path)
sys.path.append("/blue/bartlett/z.windom/LGMRES/bindings")
sys.path.append("/home/z.windom/.conda/envs/p4dev/lib/python3.8/site-packages/pyamg")
print(sys.path)
# import LGMRES
import pyamg


from random import *


def linearEqSolnPRECOND(
    invBOOL,
    H,
    H0,
    Phi,
    HPhi,
    guessE,
    Hdim,
    PspaceDim,
    b,
    normResid,
    maxiter,
    matrixVecMultiplies,
):
    # must construct A == E-QH0 == E - H0 + PH0
    ## choose different guess p

    E = guessE
    projP = np.zeros((Hdim, Hdim))
    print("length of full P space: ", len(Phi))
    if len(Phi) == Hdim:
        Phi_p = np.reshape(Phi, (Hdim, 1))
        projP = projP + Phi_p @ Phi_p.T
    else:
        for i in range(len(Phi)):
            p_i = Phi.popleft()
            p_i = p_i / np.linalg.norm(p_i)
            Phi_p = np.reshape(p_i, (Hdim, 1))
            print("first 5 of p_i", Phi_p[:5, 0])
            projP = projP + Phi_p @ Phi_p.T

    projQ = np.eye(Hdim) - projP
    if invBOOL == "QHQ":
        diffH = H - np.eye(Hdim) * guessE
        A = (projQ @ diffH) @ projQ
        # print(guessE,np.diag(np.diag(H)),np.dot(Phi[:],Phi[:]))
        print("residual RHS: ", b[:5])
        print("@linearEqSolnPRECOND: Solving for full QHQ ... \n")
        # precondA=np.linalg.inv(np.diag(np.diag(H))-guessE*np.eye(Hdim))

        # x=LGMRES.solve(A,b,M=precondA,tol=10**-5,maxiter=1,inner_m=5,outer_k=3)
        x00 = np.zeros((Hdim,))  # np.random.uniform(size=(Hdim))*(10**-10/Hdim)
        x00[0] = 1.0
        # x00=np.random.rand(Hdim,)
        print("overlap of r: ", np.linalg.norm(b))
        # xx=LGMRES.solve(A,b,tol=normResid*10**-4,x0=x00,maxiter=1,inner_m=15,outer_k=1)
        print("CG tol: ", normResid * 1e-5)
        M = np.zeros((Hdim, Hdim))
        M = np.diag(np.diag(A))
        for zzz in range(Hdim):
            xxx = M[zzz][zzz]
            if abs(xxx) < 0.0001:
                xxx = math.copysign(0.0001, xxx)
            M[zzz, zzz] = 1.0 / xxx

            # inner_m=3, outer_k=3,
        # xx,info=lin.lgmres(A,-b,maxiter=20,x0=x00,tol=normResid*1e-4,M=M)
        # xx=LGMRES.solve(A,-b,tol=normResid*10**-4,x0=x00,maxiter=20)
        # print('info: ',info)

        # xx,info=lin.lgmres(A,-b,maxiter=20,x0=x00,tol=normResid*1e-5)
        # 10 matrix-vector multiplies inside inner_m
        H0rinv = np.eye(Hdim) * guessE - H0
        for zzz in range(Hdim):
            xxx = H0rinv[zzz][zzz]
            if abs(xxx) < 0.0001:
                xxx = math.copysign(0.0001, xxx)
            H0rinv[zzz, zzz] = 1.0 / xxx

        x0 = H0rinv @ b

        xx = LGMRES.solve(
            A, -b, maxiter=3, tol=normResid * 1e-2, outer_k=3, inner_m=5, M=M, x0=x0
        )
        # print('lgmres info: ',info)
        print(
            "Checking (E-QH0Q)*x == b aka (E-QH0Q)|r>=|psi>",
            np.allclose(A @ xx, -b[:, 0], rtol=10**-4, atol=10**-4),
            np.linalg.norm(xx),
            np.linalg.norm(b),
        )
        print("solution vector: ", xx[:5])
        return xx, matrixVecMultiplies
    else:
        print("using Qh0Q")
        diffH = H0 - np.eye(Hdim) * guessE
        print(H0[:4, :4])
        # QH0Q=(projQ@H0)@projQ
        # diffH=np.eye(Hdim)*guessE-QH0Q#H0
        # A=(projQ@diffH)@projQ
        A = diffH

        # precondA=np.linalg.inv(np.diag(np.diag(A)))
        # x=LGMRES.solve(A,b,tol=1e-8,maxiter=200)
        # print('Checking (E-QH0Q)*x == b aka (E-QH0Q)|r>=|psi>',np.allclose(A@x,b[:,0],rtol=10**-4,atol=10**-4),
        #      np.linalg.norm(x),np.linalg.norm(b))
        # xx, info = gmres(A=A, b=b,tol=1e-10,M=np.linalg.inv(H0),restart=35,maxiter=40)
        x00 = np.zeros((Hdim,))  # np.random.uniform(size=(Hdim))*(10**-10/Hdim)
        x00[0] = 1.0
        # x00=np.random.rand(Hdim,)
        precon = np.linalg.inv(diffH)
        print("lmgres tol: ", normResid * 1e-5)

        xx, info = lin.lgmres(A, -b, maxiter=40, tol=1e-12, atol=1e-12)
        print("output of lgmres: ", xx[:5])
        print("Info of scipy gmres: ", info)
        print(
            "Checking (E-QH0Q)*x == b aka (E-QH0Q)|r>=|psi>",
            np.allclose(A @ xx, b[:, 0], rtol=10**-4, atol=10**-4),
            np.linalg.norm(xx),
            np.linalg.norm(b),
        )

        print("leaving linearEqSol", type(xx), type(matrixVecMultiplies))
        return xx, matrixVecMultiplies


def GetPrecon_A(Phi, H_Phi, PspaceDim, H, Hdim, guessE):
    Precon_A = np.zeros((Hdim,))
    for i in range(Hdim):
        Precon_A[i,] = (
            Precon_A[
                i,
            ]
            + guessE
            - H[i, i]
        )
        Precon_A[i,] = (
            Precon_A[
                i,
            ]
            + np.dot(Phi[i, :], Phi[i, :]) * H[i, i]
        )
        Precon_A[i,] = Precon_A[
            i,
        ] + np.dot(H_Phi[i, :], Phi[i, :])
        Precon_A[i,] = (
            Precon_A[
                i,
            ]
            - (np.dot(Phi[i, :], Phi[i, :]) ** 4) * H[i, i]
        )
    finalPrecond_A = np.diag(np.diag(Precon_A))

    projP = Phi @ Phi.T
    testPrecon = (
        -H + guessE * np.eye(Hdim) - (projP @ H) @ projP + projP @ H + H @ projP
    )
    testdiag = np.diag(testPrecon)
    np.reshape(Precon_A, (Hdim, 1))
    print(Precon_A)
    print(testdiag)
    # assert np.allclose(Precon_A,testdiag)
    return np.linalg.inv(np.diag(finalPrecond_A))


# Takes the PspaceDim most recent Phi==R, HPhi==HR vectors to define a
# new Q space through (I-P). Returns the matrix-vector multiply
# (QHQ - E)|psi>
# Intended for use inside preconditioned CG or GMRES algorithm
def Get_matmulT0guessPsi(Phi, H_Phi, PspaceDim, H, Hdim, guessE, guessPsi):
    HPsi = H @ guessPsi  # H|Psi>
    EPsi = guessE * guessPsi  # E|Psi>

    # Construct PH|Psi> & HP|Psi>
    PH_Psi = np.zeros((matrixDim,))
    HP_Psi = np.zeros((matrixDim,))
    for i in range(PspaceDim):
        PH_Psi = PH_Psi + Phi[:, i] * (np.dot(Phi[:, i], HPsi))
        HP_Psi = HP_Psi + H_Phi[:, i] * (np.dot(Phi[:, i], guessPsi))

    # Construct PHP|Psi>
    PHP_Psi = np.zeros((matrixDim,))
    for j in range(PspaceDim):
        for i in range(PspaceDim):
            PHP_Psi = PHP_Psi + Phi[:, j] * (np.dot(Phi[:, j], H_Phi[:, i])) * (
                np.dot(Phi[:, i], guessPsi)
            )

    return HPsi - EPsi + PHP_Psi - PH_Psi - HP_Psi


def addRecursiveCorrections(
    R, matrixDim, PTorder, V, H0, theta, target=0, H0def="diag"
):
    finalR = R
    tmpR = R
    p = R
    print("size of R: ", np.shape(R))
    for i in range(PTorder):
        tmpR = V @ tmpR
        print("size of tmpR: ", np.shape(tmpR))
        partOne_R = SimpleT0(tmpR, theta, H0, matrixDim, 1, target, H0def)

        partTwo = SimpleT0(p, theta, H0, matrixDim, 1, target, H0def)
        scalar = p.T @ tmpR
        partTwo_R = scalar * partTwo

        tmpR = partOne_R - partTwo_R
        finalR = finalR + tmpR
        print("shape of final R: ", np.shape(finalR))
    return finalR


def Get_CorrVec(R, residual, theta, H0, matrixDim, i, target=0, H0def="diag"):
    print("@Get_CorrVec: inserting into position ", i, "of R vec")
    print("@Get_CorrVec: Using theta ", theta)
    if H0def == "diag" or H0def == "PHP":
        R[:, i] = SimpleT0(residual, theta, H0, matrixDim, 1, target, H0def)
    else:
        print("Do not have this H0def yet built into Get_CorrVec")
        sys.exit()
    print("@Get_CorrVec: Printing T0|r> \n")
    print(R[:30, i])
    print("")
    return R


def SimpleT0(residual, theta, H0, matrixDim, blockDim, target=0, H0def="diag"):
    corrVec = np.zeros((matrixDim,))
    QH0Q = np.zeros((matrixDim, matrixDim))
    print("shapes: ", np.shape(corrVec), np.shape(residual))
    if H0def == "diag":
        QH0Q = H0
    elif H0def == "PHP":
        print("continue")

    for zzz in range(matrixDim):
        xxx = theta - QH0Q[zzz][zzz]
        # xxx=-1.0*xxx
        # print('shape of theta: ',np.shape(theta),np.shape(QH0Q))
        if abs(xxx) < 0.0001:
            xxx = math.copysign(0.0001, xxx)
        corrVec[zzz,] = (
            residual[zzz] / xxx
        )

    return corrVec


# Check if convergence has been reached by determining if the normed residual
# has fallen below some tolerance, TOL.
def checkConvergence(normResid, TOL, i, root, solNum=0):
    if normResid < TOL:
        print()
        print("*******************************")
        print("*******************************")
        print("Converged a root !!!!!")
        print()
        print("Converged root: ", solNum)
        print("Converged in ", i, " matrix-vector multiplies!!")
        print("Residual Norm at convergence: ", normResid)
        print("Root value at convergence: ", root)
        print()
        print()
        print("*******************************")
        print("*******************************")
        return True
    else:
        False


# After a root,vector has converged the expansion subspace vectors need to be removed so that
# a subsequent iteration can take place. This routine prunes the useless subspace vectors and stores the
# most recently found eigenvector in addition to an initial guess vector targeting the next lowest solution.
# Returns the revised expansion subspace, 'R', and the matrix storing its contraction with the Hamiltonian, 'HR'
def Get_solnSpace(
    H,
    R,
    HR,
    rootTotal,
    rootTarget,
    target,
    vecs,
    i,
    Hdim,
    lowestElements,
    seedUnitVecGuess=False,
):
    if seedUnitVecGuess == False:
        tmp = np.zeros((Hdim,))
        tmp1 = np.zeros((Hdim,))
        tmp = R[:, : i + target] @ vecs[:, target]
        tmp1 = R[:, : i + target] @ vecs[:, target + 1]
        print("test eval: ", (tmp.T @ H) @ tmp)
        print("new eval: ", (tmp1.T @ H) @ tmp1)
        R[:, target] = tmp
        R[:, target + 1 :] = 0.000000
        R[:, target + 1] = tmp1
        tmp = 0.00
        tmp1 = 0.00
        tmp = HR[:, : i + target] @ vecs[:, target]
        tmp1 = HR[:, : i + target] @ vecs[:, target + 1]
        HR[:, target + 1 :] = 0.0000
        HR[:, target] = tmp
        HR[:, target + 1] = tmp1
    else:
        tmp = np.zeros((Hdim,))
        tmp1 = np.zeros((Hdim,))
        tmp = R[:, : i + target] @ vecs[:, rootTarget]
        # tmp1=np.zeros((Hdim,1))
        tmp1[
            lowestElements[target + 1],
        ] = 1.0
        print("old converged eval: ", (tmp.T @ H) @ tmp)
        print("next guess to eval: ", (tmp1.T @ H) @ tmp1)

        R[:, target] = tmp
        R[:, target + 1 :] = 0.000000
        R[:, target + 1] = tmp1
        R = GramSchmidt(R, matrixDim, target + 2)
        print("next guess to eval: ", (R[:, target + 1].T @ H) @ R[:, target + 1])

        tmp = 0.0
        tmp1 = 0.0
        tmp = HR[:, : i + target] @ vecs[:, rootTarget]
        tmp1 = H @ R[:, target + 1]
        HR[:, target] = tmp
        HR[:, target + 1 :] = 0.0000

        HR[:, target + 1] = tmp1
        print("Test to make sure R and HR is emptied appropriately")
        print(R[:, : target + 3])
        print(HR[:, : target + 3])
        print(
            "matrix comparing HR and R magnitudes: \n",
            HR[:, : target + 2].T @ R[:, : target + 2],
        )
        print(np.linalg.norm(HR[:, target + 1]))

    return R, HR


# Calculates and returns the Davidson residual vector and its norm
def Get_residual(H, R, HR, vecs, i, matrixDim, blockDim, p, theta, target=0):
    residual = np.zeros((matrixDim, blockDim))
    # print("@Get_residual: value of i: ",i)
    # print("@Get_residual: value of target: ",target)
    if i == 0:
        print(np.shape(HR), np.shape(HR[:, 0 + target]))
        residual[:, 0] = HR[:, 0 + target] - theta * R[:, 0 + target]
    else:
        residual[:, 0] = HR[:, :i] @ vecs[:, target] - theta * (
            R[:, :i] @ vecs[:, target]
        )

    normResid = np.linalg.norm(residual)
    print("norm resid", normResid)
    # print('residual ',residual)
    return residual, normResid


# Determines if a lower energy solution than the one currently being targeted has appeared in the subspace.
# If so, adjust the target index accordingly.
def CheckSubspaceCollapse(convRoots, RitzRoots, newTarget):
    cRoots = np.sort(convRoots)
    print("convRoots: ", cRoots)
    for i in range(len(convRoots)):
        tmp = abs(abs(cRoots[i]) - abs(RitzRoots[i]))
        print("value of tmp: ", tmp)
        if tmp >= 0.0001:
            newTarget = i
            break

    return newTarget


# Collapses the expansion subspace to contain only 'numVecs_keep' # of guess vectors.
# Also modifies the contraction of Hamiltonian with expansion vectors, 'HR', accordingly.
def Do_SubSpaceCollapse(
    H, R, HR, Tvecs, currentDim, numVecs_keep, rootsFound, matrixDim
):  # rootsFound==target is the current root we are trying to target
    R_queue = deque()
    HR_queue = deque()
    # totalSSdim=currentDim+1+rootsFound
    print("@Do_SubSpaceCollapse: current, total SS dim:", currentDim + 1)
    print(
        "BUT keeping latest ",
        numVecs_keep,
        " vectors in the S.S and keeping: ",
        rootsFound,
        "eigenvector solutions",
    )
    print(
        "@Do_SubSpaceCollapse Does *NOT* include vectors already found; these are kept by default"
    )

    # make sure to save the eigenvector solutions already found
    if rootsFound > 0:
        for x in range(rootsFound):
            R_queue.append(R[:, x])
            HR_queue.append(HR[:, x])

    cDim = currentDim + rootsFound
    print("cDim is:", cDim, np.shape(Tvecs))
    for xx in range(numVecs_keep):
        indx = numVecs_keep - 1 - xx + rootsFound
        print("Keep indx: ", indx)
        trans_R = R[:, : cDim + 1] @ Tvecs[:, indx]
        trans_HR = HR[:, : cDim + 1] @ Tvecs[:, indx]
        R_queue.append(trans_R)
        HR_queue.append(trans_HR)
        print("TEST <R|H|R>: ", np.dot(trans_R, trans_HR), np.dot(trans_HR, trans_R))
    # store the most recent subspace vectors
    # for xx in range(numVecs_keep-1,-1,-1):
    #    print('saving vec in dim: ',currentDim-xx)
    #    R_queue.append(R[:,currentDim-xx])
    #    HR_queue.append(HR[:,currentDim-xx])

    totLength = len(R_queue)
    # copy the saved eigevector solns. & the most recent vectors in queue to R, HR vectors
    for jj in range(len(R_queue)):
        R[:, jj] = R_queue.popleft()
        HR[:, jj] = HR_queue.popleft()
    print("jj is: ", jj)
    print("Length of queue ", len(R_queue))
    R[:, jj + 1 :] = 0.0
    HR[:, jj + 1 :] = 0.0
    print("Final R: \n\n", R[:30, : jj + 1])
    R = GramSchmidt(R, matrixDim, jj + 3)

    for kk in range(totLength):
        HR[:, kk] = H @ R[:, kk]
    # HR=GramSchmidt(HR,matrixDim,jj+2)
    tmps = R[:, : jj + 1].T @ HR[:, : jj + 1]
    print("<R|H|R>:")
    print(tmps)
    roots, evecs = np.linalg.eig(tmps)
    print("sorted roots: ", np.sort(roots)[:6])
    print("other <R|H|R>.T \n", HR[:, : jj + 1].T @ R[:, : jj + 1])
    targetIndx = numVecs_keep + rootsFound
    print("Returning ending index: ", rootsFound + numVecs_keep - 1)
    return R, HR, jj - rootsFound  # rootsFound+numVecs_keep-1 #


# Determine if the program has enough subspace vectors to warrant a collapse
def Query_SubSpaceCollapse(i, maxSSdim, rootsFound):
    totalSSdim = maxSSdim + rootsFound
    return i + 1 + rootsFound == totalSSdim


def Initialize_LinearEqnSolv(
    invBOOL,
    Amat,
    H0,
    R,
    HR,
    theta_p,
    matrixDim,
    PspaceDim,
    residual,
    normResid,
    matrixVecMultiplies,
):
    print("Inverting the full resolvent operation", invBOOL)
    solnR, matrixVecMultiplies = linearEqSolnPRECOND(
        invBOOL,
        Amat,
        H0,
        R,
        HR,
        theta_p,
        matrixDim,
        PspaceDim,
        residual,
        normResid,
        200,
        matrixVecMultiplies,
    )
    print("returning from Initialize_linear eq solv")
    print(type(solnR), type(matrixVecMultiplies))
    print(solnR[:5], matrixVecMultiplies)
    return solnR, matrixVecMultiplies


#    if invBOOL=="QH0Q":
#        print("Inverting  (E-QH0Q)^-1 at no cost... ")
#        solnR,matrixVecMultiplies=linearEqSolnPRECOND(invBOOL,Amat,H0,R,HR,\
#                                             theta_p,matrixDim,PspaceDim,residual,CGtol,\
#                                             200,matrixVecMultiplies)
#    elif invBOOL=="QHQ":
#        print("Inverting the full resolvent operation, (E-QHQ)^-1, approximately...")
#        solnR,matrixVecMultiplies=linearEqSolnPRECOND(invBOOL,Amat,H0,R,HR,\
#                                             theta_p,matrixDim,PspaceDim,residual,CGtol,\
#                                             200,matrixVecMultiplies)


# Program: General-purpose, single-vector Davidson-based iterative diagonlizer constructed using BW-PT
#          Can be used to perform the traditional Davidson algorithm and has subspace collapse capabilities.
# .         Perhaps most importantly, this program extends the traditional Davidson algorithm for use on atypical
#          (non-diagonally dominant, non-sparse) matrices using the principles of Brillouin-Wigner Perturbation Theory (BW-PT)
#          As a result, this software is equipped to (attempt to) iteratively invert the resolvent operator, (E-QHQ)^-1, and/or
#          its first order approximation, (E-QH0Q)^-1 using the LGMRES algorithm.
# Constructed by: Zachary Wayne Windom 6/5/2022
#
# *****  INPUT:  *****
#
# Amat: The matrix to be iteratively diagonalized; will refer to this as the Hamiltonian, H, equivalently
# matrixDim: Column rank of Amat
# blockDim: Dimension of the number of starting vectors in |p>; restricted to be 1 for now.
# outputFileName: name of output file to store convergence information
# p: The initial guess seeding the start of the algorithm
# theta: The estimated eigenvalue
# TOL: Convergence tolerance defined by the norm of the residual vector
# maxItr: Max. number of iterations per root
# solnCount:
# H0def: The definition of zeroth order Hamiltonian. Currently only supports Hdiag. Limited support of PHP==H0
# highOresolvent: Dictionary storing the microiteration in which inversion of the resolvent operator is required
# invBOOL: Determines whether the approximation to the full resolvent operator, (E-QH0Q)^-1, is inverted, or an attempt is made
#               to invert the full resolvent operator, (E-QHQ)^-1, in a limited number of iterations.
# numSSVecs_keep: Number of recent SS vectors to keep
# maxSSvecs: Max. allowed number of vectors in the SS (not including prior solution vectors)
#
# ***** OUTPUT: *****
#
# matVecMult: Number of matrix-vector multiplies required to achieve convergence per root
# theta: The estimated eigenvalue(s) at each increment of matVecMult
# resid: An array of the residual vectors corresponding to each root


def Get_BWPT_Expansion(
    residual, p, theta_p, H0, H, V, matrixDim, matmuls, BWPTorder=1, target=0
):
    # calculate T0|r> & T0|p>
    # if matmuls==1:
    # print('modifying p')
    # p=p+np.random.rand(matrixDim,1)
    # p=p/np.linalg.norm(p)

    print("shapes of residual and p: ", np.shape(residual), np.shape(p))
    T0_R = SimpleT0(residual, theta_p, H0, matrixDim, 1, target=0, H0def="diag")
    T0_p = SimpleT0(p, theta_p, H0, matrixDim, 1, target=0, H0def="diag")
    Hp = H @ p
    T0_HP = SimpleT0(Hp, theta_p, H0, matrixDim, 1, target=0, H0def="diag")

    print(
        "shape of T0s: ",
        np.shape(T0_R),
        np.shape(T0_p),
        np.linalg.norm(T0_p),
        np.linalg.norm(T0_R),
    )

    if BWPTorder > 1:
        VT0_p = V @ T0_p
        VT0_R = V @ T0_R

        T0_VT0_p = SimpleT0(VT0_p, theta_p, H0, matrixDim, 1, target=0, H0def="diag")
        T0_VT0_R = SimpleT0(VT0_R, theta_p, H0, matrixDim, 1, target=0, H0def="diag")

        alpha = (p[:, 0].T @ T0_R + p[:, 0].T @ T0_VT0_R) / (
            p[:, 0].T @ T0_p + p[:, 0].T @ T0_VT0_p
        )
        print("alpha: ", alpha)
        corrVec = T0_R + T0_VT0_R - alpha * (T0_p + T0_VT0_p)
        print(
            "@Get_BWPT_Expansion: Checking if <p|psi>==0:",
            p.T[
                :,
            ]
            @ corrVec,
        )
        matmuls = matmuls + 2
    else:
        print("shapes of alpha var: ", np.shape(p), np.shape(T0_p), np.shape(T0_R))
        print("overlap <p|r>:", p[:, 0].T @ residual)
        # alpha=(np.dot(p[:,0],T0_R))/(np.dot(p[:,0],T0_p))
        # print('alpha: ',alpha, np.shape(alpha))
        # corrVec=alpha*T0_p -T0_R

        scalars = theta_p + (Hp[:].T @ T0_HP)  # + (T0_HP[:].T@V)@T0_HP[:]
        corrVec = T0_R - scalars * T0_p
        print("@Get_BWPT_Expansion: Checking if <p|psi>==0:", p.T[:] @ corrVec)
        matmuls = matmuls

    print("shape of corrVec:", np.shape(corrVec))
    return corrVec, matmuls


def Get_OlsenVec(new_p,residual,H0,theta_p,matrixDim,H):
    print('check <p|H|p> is correct:',(new_p.T@H)@new_p == theta_p)
    # get (H_0 - E_0)^-1 |r>
    shifted_resid=-1.0*SimpleT0(residual,theta_p,H0,matrixDim,1)

    # get (H_0 - E_0)^-1 |p>
    shifted_p=-1.0*SimpleT0(new_p,theta_p,H0,matrixDim,1)

    scalar_num=new_p.T@shifted_resid
    scalar_denom=new_p.T@shifted_p
    scalar=scalar_num/scalar_denom
    print('scalar is: ', scalar)
    sys.exit()

    # get (E - H0)^-1* scalar * |p>
    olsenVec=SimpleT0(scalar*new_p, theta_p,H0,matrixDim,1)
    return -1.0*olsenVec

def Run_DavidsonBWPT(
    Amat,
    matrixDim,
    blockDim,
    outputFileName,
    p=np.asarray([]),
    theta=None,
    TOL=10**-6,
    maxItr=20,
    solnCount=1,
    H0def="diag",
    highOresolvent=OrderedDict(),
    invBOOL="QHQ",
    numSSVecs_keep=1,
    maxSSvecs=5,
    iterH0def="dynamic_QH0Q",
    spectrum="lowest",
):  #'static_QH0Q'):
    # Initialize starting guess, |p>, guess eval, theta, and the sorted array of diagonal elements.
    # Then partition the full H=H0+V, store |p> into 'R', contract 'R' with the full H, compute the residual.
    # This is essentially the first step of the algorithm.
    if p.size == 0:
        p, theta, lowestElements = Get_p(Amat, matrixDim, blockDim, True)  # ,iterH0def)
    elif theta == None:
        theta = (p.T @ Amat) @ p
    # else:
    #    theta=(p.T@Amat)@p
    H0, V = Get_H0(Amat, p, matrixDim, blockDim, H0def)
    print("Example H0: ", H0[:4, :4])
    # print(p[350,0])
    R, HR = Get_InitialSpace(Amat, p, theta, matrixDim, blockDim, maxItr)
    testOUT = (R[:, 0].T @ Amat) @ R[:, 0]
    # print('Starting HR[:,0]:',np.allclose(testOUT,np.dot(R[:,0],HR[:,0])))
    residRecord = []
    convRoots = []
    thetaRecord = []
    vecs = []
    theta_p = theta
    print("Initial Eval: ", theta_p)
    print(R[:, 0].T @ HR[:, 0])

    new_p = p
    overallMatMul = []
    for target in range(solnCount):
        matrixVecMultiplies = 1
        rootMatMul = []
        rootMatMul.append(matrixVecMultiplies)

        residList = []
        thetaList = []
        print()
        print("\n\n\n")
        print("***************************")
        print("Starting the solution for root: ", target)
        print()
        RootTarget = target
        residual, normResid = Get_residual(
            Amat, R, HR, vecs, 0, matrixDim, blockDim, p, theta_p, RootTarget
        )
        residList.append(normResid)
        print("Initial guess theta: ", theta_p)
        thetaList.append(theta_p)
        if target == 0:
            SSevalList = []
            SSevalList.append(np.asarray([theta_p]))
        i = 0

        ritzVecs_Set = deque()
        ritzVecs_Set.append(p)
        for j in range(maxItr):
            print("\n\n\n\n Starting iteration: ", j)

            # Check to see if SS collapse is needed
            numSSVecs_keep = 200  # number of recent SS vectors to keep
            maxSSvecs = 5000  # Max. allowed number of vectors in the SS (not including prior solution vectors)
            if Query_SubSpaceCollapse(i, maxSSvecs, target):
                R, HR, i = Do_SubSpaceCollapse(
                    Amat, R, HR, vecs, i, numSSVecs_keep, target, matrixDim
                )

            # Check to see if inverting the full/partial resolvent operator, T0, is needed.
            # If so, call external LGMRES to solve x=A^-1b.
            # Otherwise, build the Davidson-like correction vector
            # if i in highOresolvent.keys() and normResid>10**-4:# and target==0:
            if iterH0def == "dynamic_QH0Q":  # and (i>0 and i<10) and normResid>10**-5:
                print("Inverting  (E-QH0Q)^-1 **OR** the full (E-QHQ)^-1")
                PspaceDim = 0
                Sindx = target + i - PspaceDim  # sending in latest 3 vectors in R
                Eindx = target + i + 1
                CGtol = 10**-7
                invBOOL = "QH0Q"
                # H0=np.diag(np.diag(Amat))
                print("sending r 348", R[348, 0])
                gap_estimate = 0.1  # theta[-1]/theta[0] if j>0 else theta_p
                print("gap estimate: ", gap_estimate)
                sigma = (
                    theta_p - gap_estimate
                )  # gap_estimate is estimate ratio of evalMAX/evalMIN
                ritzVecs_SetBKUP = ritzVecs_Set.copy()
                R[:, target + i + 1], matrixVecMultiplies = Initialize_LinearEqnSolv(
                    invBOOL,
                    Amat,
                    H0,
                    new_p,
                    HR[:, target + i],
                    sigma,
                    matrixDim,
                    PspaceDim,
                    residual,
                    normResid,
                    matrixVecMultiplies,
                )
            elif iterH0def == "static_QH0Q":
                R[:, target + i + 1], matrixVecMultiplies = Get_BWPT_Expansion(
                    residual,
                    new_p,
                    theta_p,
                    H0,
                    Amat,
                    V,
                    matrixDim,
                    matrixVecMultiplies,
                    BWPTorder=1,
                    target=target,
                )
                # R[:,target+i+1],matrixVecMultiplies=Get_BWPT_Expansion(residual,p,theta_p,H0,Amat,V,matrixDim,matrixVecMultiplies,
                #                                                       BWPTorder=2,target=0)

            elif (
                H0def == "TriDiag"
                or H0def == "dynamicH0Diag"
                or H0def == "BiDiag"
                or H0def == "QuadDiag"
            ):
                print("Inverting the H0 approximation using ...", H0def)
                sigma = theta_p
                PspaceDim = 0
                print("RHS: ", residual[:5])
                R[:, target + i + 1], matrixVecMultiplies = Initialize_LinearEqnSolv(
                    invBOOL,
                    Amat,
                    H0,
                    new_p,
                    HR[:, target + i],
                    sigma,
                    matrixDim,
                    PspaceDim,
                    residual,
                    normResid,
                    matrixVecMultiplies,
                )
            else:
                print("Norm of residual in main loop: ", np.linalg.norm(residual))
                R = Get_CorrVec(
                    R, residual, theta_p, H0, matrixDim, target + i + 1, 0, H0def
                )
                if iterH0def == 'Olsen':
                    olsenCorrVec=Get_OlsenVec(new_p,residual,H0,theta_p,matrixDim,H)
                    R[:,target+i+1]+=olsenCorrVec

            # Orthonormalize the subspace vectors in R, contract the latest R-vector
            # with the Hamiltonian, then compute and diagonalize <R|H|R>
            R = GramSchmidt(R, matrixDim, target + i + 2)
            HR[:, target + i + 1] = Amat @ R[:, target + i + 1]
            matrixVecMultiplies = matrixVecMultiplies + 1
            rootMatMul.append(matrixVecMultiplies)
            theta, vecs = computeRHR(R, HR, target + i + 1, blockDim)
            if spectrum == "highest":
                highestIndx = (-theta).argsort()[: i + 2]
                print("highest index: ", highestIndx)
                theta = theta[highestIndx]
                vecs = vecs[:, highestIndx]

            if target == 0:
                SSevalList.append(theta)

            # Check to see if a  solution exists that is lower in energy than existing solution vectors in the subspace
            # Possibly obsolete ZWW 6/5/2022
            RootTarget = CheckSubspaceCollapse(
                convRoots, theta[: target + 2], RootTarget
            )
            print("rootTarget after SS collapse", RootTarget)
            theta_p = theta[RootTarget]

            print("Ritz estimate: ", theta_p)
            thetaList.append(theta_p)

            # Calculate the residual as QH|p>, or equivalently
            # |r>=H|R>@Tvecs[] - theta_p*|R>@Tvecs[]
            residual, normResid = Get_residual(
                Amat,
                R,
                HR,
                vecs,
                target + i + 2,
                matrixDim,
                blockDim,
                p,
                theta_p,
                RootTarget,
            )
            new_p = (
                R[:, : target + i + 2] @ vecs[:, target]
            )  # this is our current Ritz vector; used in J-D procedure
            ritzVecs_Set.append(new_p)
            print("length of ritZVECS", len(ritzVecs_Set))
            residList.append(normResid)

            print("MQ is my R: ", R[:5, i + 1])

            # Check for convergence of ||<r|r>||; if true, then store the solution eigenvector into
            # 'R' variable, and setup the next macro-cycle's initial guess
            if checkConvergence(
                normResid, TOL, matrixVecMultiplies, theta_p, RootTarget
            ):
                fevecs = R[:, : target + i + 2] @ vecs[:, target]
                print("<p|H0|p>:", (fevecs.T @ H0) @ fevecs)
                fresolv = theta[target] * np.eye(matrixDim) - H0
                for xxxxx in range(matrixDim):
                    fresolv[xxxxx, xxxxx] = 1.00000 / fresolv[xxxxx, xxxxx]
                print("<r|(E-H0)^-1|p>", (residual.T @ fresolv) @ residual)
                print(
                    "total E from parts: ",
                    (residual.T @ fresolv) @ residual + (fevecs.T @ Amat) @ fevecs,
                )
                print("actual E: ", theta[target])
                convRoots.append(theta_p)
                seedUnitVecGuess = False
                rootTotal = len(convRoots) - 1
                print("total converged roots: ", rootTotal + 1)
                if solnCount != target + 1:
                    R, HR = Get_solnSpace(
                        Amat,
                        R,
                        HR,
                        rootTotal,
                        RootTarget,
                        target,
                        vecs,
                        i + 2,
                        matrixDim,
                        lowestElements,
                        seedUnitVecGuess,
                    )
                theta_p = theta[
                    target + 1
                ]  # HR[:,target+1].T@R[:,target+1]#theta[target+1]
                print("estimate of theta_p: ", theta_p)
                residRecord.append(residList)
                thetaRecord.append(thetaList)
                break

            i = i + 1
        overallMatMul.append(rootMatMul)

    if solnCount == 3:
        sortedInfo = postProcessInfo(
            outputFileName, overallMatMul, thetaRecord, residRecord
        )
        pltConvergenceInfo(
            [sortedInfo],
            pltTitle=outputFileName,
            diagStratLabels=["standard Davidson"],
            pltResid=True,
        )

    return overallMatMul, thetaRecord, residRecord, SSevalList, new_p
