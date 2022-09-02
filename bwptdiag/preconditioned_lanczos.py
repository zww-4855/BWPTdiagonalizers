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
from bwptdiag.lanczos import Run_Lanczos


def cholesky(A):
    """Performs a Cholesky decomposition of A, which must
    be a symmetric and positive definite matrix. The function
    returns the lower variant triangular matrix, L."""
    n = len(A)

    # Create zero matrix for L
    L = [[0.0] * n for i in range(n)]

    # Perform the Cholesky decomposition
    for i in range(n):
        for k in range(i + 1):
            tmp_sum = sum(L[i][j] * L[k][j] for j in range(k))
            if abs(A[i][i]) < 0.0001:
                A[i][i] = math.copysign(0.0001, A[i][i])

            if i == k:  # Diagonal elements
                # LaTeX: l_{kk} = \sqrt{ a_{kk} - \sum^{k-1}_{j=1} l^2_{kj}}
                L[i][k] = np.sqrt(A[i][i] - tmp_sum)
            else:
                # LaTeX: l_{ik} = \frac{1}{l_{kk}} \left( a_{ik} - \sum^{k-1}_{j=1} l_{ij} l_{kj} \right)
                L[i][k] = 1.0 / L[k][k] * (A[i][k] - tmp_sum)
    return L


def Get_Mk(
    H,
    H0,
    E,
    p,
    dim,
    QHQapprox="diag",
    customMk=np.array([]),
    i=1,
    rootTarget=0,
    convgVecs=np.array([]),
    shift=0.25,
):
    if customMk.size != 0:
        if i == 1:
            return customMk
        else:
            M = np.arange(10.1, 110.1, 0.1)
            print("custom Mk: \n", np.diag(M) - E * np.eye(dim))
            return np.diag(M) - E * np.eye(dim)

    if QHQapprox == "diag" or QHQapprox == "updateH0":
        if E > max(np.diag(H0)):
            Mk = H0
        else:
            Mk = H0 - E * np.eye(dim)  # +np.diag(np.diag(shift))
            print("minimum element of Mk: ", np.amin(Mk))
            # Mk=Mk+np.amin(Mk)*np.eye(dim)

    elif QHQapprox == "constH0shift":
        Mk = H0 - (shift * np.eye(dim) + rootTarget * 0.05)

    elif QHQapprox == "moment":
        Mk = E * np.eye(dim)

    elif QHQapprox == "QH0Q":
        Q = np.eye(dim) - p @ p.T
        Mk = (Q @ H0) @ Q - E * np.eye(dim)
    elif QHQapprox == "QHQ":
        Q = np.eye(dim) - p @ p.T
        Mk = (Q @ H) @ Q - E * np.eye(dim)
    else:
        print("@Get_Mk: QHQapprox. not implemented!! Exiting...")
        sys.exit()

    print("@Get_Mk: Mk : \n", Mk[:5, :5])
    return Mk


def Custom_cholesky(A, Adim, numberOfAddRows):
    """Performs a Cholesky decomposition of A, which must
    be a symmetric and positive definite matrix. The function
    returns the lower variant triangular matrix, L."""
    L = np.zeros((Adim, Adim))
    # Perform the Cholesky decomposition
    for i in range(Adim):
        j_start = 0 if 0 > i - numberOfAddRows else i - numberOfAddRows
        # print('jstart: ',j_start)
        for j in range(j_start, i + 1):
            summed = 0.0
            k_start = 0 if 0 > j - numberOfAddRows - 1 else j - numberOfAddRows - 1
            for k in range(k_start, j):
                summed = summed + L[i, k] * L[j, k]

            if abs(A[i][i]) < 0.0001:
                A[i][i] = math.copysign(0.0001, A[i][i])

            if i == j:
                L[i, j] = np.sqrt(A[i, i] - summed)
            else:
                L[i, j] = (1.0 / L[j, j]) * (A[i, j] - summed)

    return L


def Get_CholeskyFactMk(Mk, dim, QHQapprox="diag", H0def="diag"):
    if (
        QHQapprox == "diag"
        or QHQapprox == "moment"
        or QHQapprox == "updateH0"
        or QHQapprox == "constH0shift"
    ):
        if H0def == "diag":
            # print('inside moment exp:',QHQapprox)
            # sys.exit()
            Lk = np.zeros((dim, dim))
            for i in range(dim):
                xxx = Mk[i, i]
                if abs(xxx) < 0.0001:
                    xxx = math.copysign(0.0001, xxx)
                    Lk[i, i] = abs(xxx) ** (0.50000000)
                else:
                    Lk[i, i] = np.sqrt(Mk[i, i])  # abs(Mk[i,i])**(0.50000000)
        else:
            print("@Get_CholeskyFactMk: working with triDiagonal H0 \n\n")
            # u,v =np.linalg.eig(Mk)
            # print('sorted eigenvalues of Mk:')
            # print(np.sort(u)[:5])
            Lk = np.zeros((dim, dim))
            # Lk=cholesky(Mk)
            Lk = Custom_cholesky(Mk, dim, 3)
            Lk = np.asarray(Lk)
            print("type of Lk: ", type(Lk))
            # Lk=np.linalg.cholesky(Mk)
        # assert np.allclose(np.linalg.cholesky(Mk),Lk)
    else:

        Lk = np.linalg.cholesky(Mk)

    return Lk


def Get_TransMatWk(
    H,
    dim,
    E,
    Lk,
    QHQapprox="diag",
    H0def="diag",
    rootTarget=0,
    convgVecs=np.array([]),
    shift=0,
):
    if QHQapprox == "moment":
        Wk = E * np.eye(dim) - H  # -- targets high energy solns.
    else:
        Wk = H - E * np.eye(dim)

    if (
        QHQapprox == "diag"
        or QHQapprox == "moment"
        or QHQapprox == "updateH0"
        or QHQapprox == "constH0shift"
        and H0def == "diag"
    ):
        # print('inside invert Lk: Get_TransMat')
        # sys.exit()
        invLk = np.zeros((dim, dim))
        for i in range(dim):
            invLk[i, i] = 1.0 / Lk[i, i]
        Wk = (invLk @ Wk) @ invLk.T
    #        if QHQapprox=="moment":
    #            invE=np.zeros((dim,dim))
    #            for i in range(dim):
    #                invE[i,i]=1.0/E
    #            Wk=invE@Wk

    else:
        print("@Get_TransMatWk: working inside inverting triDiag H0")
        invLk = np.linalg.inv(Lk)
        Wk = (invLk @ Wk) @ invLk.T
    return Wk


def Get_UpdatedHO_Def(H0, H, p, H0def, matrixDim):
    ovrlap = (p.T @ H) @ p
    if H0def == "updateFullH0":
        H0 = H0 + ovrlap * (p @ p.T)
        return H0
    elif H0def == "updateDiagH0":
        for i in range(matrixDim):
            H0[i, i] = H0[i, i] + ovrlap * (p[i]) ** 2
        return H0


"""
convgVecs - list of converged eigenvectors
convgThetas-list of converged eigenvalues
Amat - matrix we are diagonalizing
dim - dimension of Amat
shiftList - list of all the prior, converged shifts we once converged 
nextItrTheta - estimate of next iteration's eigenvalue
rootTarget - current root we are targeting, starting from root 0

Returns the calculated shift needed to keep the matrix positive definite

shift >= (theta_k - Amat)*vec[k-1]@vec[k-1].T -    }

"""


def Get_MatrixShifts(
    convgVecs, convgThetas, Amat, dim, shiftList, nextItrTheta, rootTarget
):
    tmp1 = nextItrTheta * np.eye(dim) - Amat
    tmp1 = tmp1 @ (convgVecs[rootTarget - 1] @ convgVecs[rootTarget - 1].T)
    tmp2 = 0.0
    for i in range(rootTarget - 1):
        ovrlap = np.dot(convgVecs[i], convgVecs[i + 1])
        tmp2 = tmp2 - convgThetas[i] * ovrlap * (convgVecs[i] @ convgVecs[i + 1].T)
    return tmp1 + tmp2 + 0.00001


def update_H0def(Amat, p, E, matrixDim, H0):
    newH0 = np.zeros((matrixDim, matrixDim))
    Hp = Amat @ p
    pH = p.T @ Amat
    shift = 0.0
    for i in range(matrixDim):
        # print(np.shape(p),np.shape(Hp),np.shape(pH))
        # print('Compare Hp vs pH element-wise: ', np.allclose(Hp[i]*p[i],p[i]*pH[i]))
        scalar = (p.T @ Amat) @ p
        newH0[i, i] = Amat[i, i] - 2.0 * Hp[i] * p[i] + scalar * p[i] ** 2
        if newH0[i, i] < E and newH0[i, i] - E < shift:
            shift = newH0[i, i] - E  # +0.01

    newH0 = newH0 - shift * np.eye(matrixDim)

    print("compare new H0 and old H0: ", np.allclose(H0, newH0))
    return newH0


# Algorithm largely follows the notation and presciption found in:
# SIAM Journal on Scientific Computing,1993 10.1137/0914037 by R. Morgan, D. Scott
def Run_PrecondLanczos(
    Amat,
    matrixDim,
    H0def="diag",
    microItr=3,
    macroItr=10,
    TOL=10**-5,
    p=np.array([]),
    QHQapprox="diag",
    PreLanczosBOOL=True,
    customMk=np.array([]),
    solnCount=1,
):
    # Initialize initial guess, |p>, eval, theta. Add it constant negative shift to eval to prevent
    # singularities when going to invert.
    blockDim = 1
    if p.size == 0:

        p, theta, lowestElements = Get_p(Amat, matrixDim, blockDim, True)
        theta = theta  # -0.0000001#0.02
    else:
        theta = (p.T @ Amat) @ p  # -0.5

    final_matmuls = []
    final_residList = []
    final_evals = []
    convgVecs = []
    convgTheta = []
    shiftMAT = np.zeros((matrixDim, matrixDim))
    shiftList = []
    for rootTarget in range(solnCount):
        print("\n\n\n\n\n\n\n\n\n")
        print("RUNNING PRECONDITIONED LANCZOS TO TARGET ROOT ", rootTarget)
        PrecondLanz_ResidList = []
        PrecondLanz_SSevalList = []
        PrecondLanz_matrixVecMultiply = []

        theta = (p.T @ Amat) @ p
        print("Starting residual: ", np.linalg.norm(Amat @ p - theta * p))
        print("Starting theta: ", theta)
        PrecondLanz_ResidList.append(np.linalg.norm(Amat @ p - theta * p))
        PrecondLanz_SSevalList.append(theta)

        print("matrix dim:", matrixDim)
        if rootTarget == 0:
            H0, V = Get_H0(Amat, p, matrixDim, blockDim, H0def)
        else:
            rootsu, bsroots = eigh(Amat)
            print("spectra of Amat before: ", np.sort(rootsu)[:5])
            Amat = (
                Amat
                + convgTheta[rootTarget - 1]
                * convgVecs[rootTarget - 1]
                @ convgVecs[rootTarget - 1].T
            )
            rootsu, bsroots = eigh(Amat)
            print("spectra of Amat after shift: ", np.sort(rootsu)[:5])
            # sys.exit()
            H0, V = Get_H0(Amat, p, matrixDim, blockDim, H0def)
            H0 = H0  # +5.000*np.eye(matrixDim)
            p, theta, lowestElements = Get_p(Amat, matrixDim, blockDim, True)
            # theta=theta-convgTheta[rootTarget-1]
            print("sorted off diagonal elements:", np.sort(np.diag(H0, k=1))[:5])
            print("sorted diagonal elements: ", np.sort(np.diag(H0, k=0))[:5])
            print(H0[:5, :5], Amat[:5, :5])
            # sys.exit()
            print("H0dim:", np.shape(H0))

        mults = 0

        for i in range(1, macroItr + 1):
            print("\n\n\n\n\n\n\n\n\n")
            print("Starting iteration: ", i)

            # Following the notation in the above work by R. Morgan et. al.
            # Calculate the inverted preconditioner, Mk, its Cholesky decomposition, Lk,
            # and then use this to similarity transform the original matrix, Amat, into Wk.
            # The latter is used as an argument into a subsequent Lanczos iteration.
            p = p / np.linalg.norm(p)
            # if convgTheta and i ==1 and rootTarget>0:
            # shiftVal=Get_MatrixShifts(convgVecs,convgTheta,Amat,matrixDim,shiftList,theta,rootTarget)
            # print('Calculated shiftValue: ',shiftVal)
            # shiftList.append(shiftVal)
            # shiftMAT=shiftMAT+convgTheta[rootTarget-1]*(convgVecs[rootTarget-1]@convgVecs[rootTarget-1].T)+0.1*np.eye(matrixDim)
            # shift=np.eye(matrixDim)*shiftVal+0.00001
            # print('Applying a shift of : ',shiftVal)

            print("minimum of H0: ", min(np.diag(H0)), "vs E: ", theta)
            Mk = Get_Mk(
                Amat,
                H0,
                theta,
                p,
                matrixDim,
                QHQapprox,
                customMk,
                i,
                rootTarget,
                convgVecs,
                shift=0.0,
            )  # Mk is approx to (H-E)
            print("mk diag", np.diag(Mk))
            Lk = Get_CholeskyFactMk(Mk, matrixDim, QHQapprox, H0def)  # Mk=Lk@Lk.T
            print("lk diag: ", np.diag(Lk))
            Wk = Get_TransMatWk(
                Amat,
                matrixDim,
                theta,
                Lk,
                QHQapprox,
                H0def,
                rootTarget,
                convgVecs,
                shift=shiftMAT,
            )  # Wk=Lk^-1@(H-E)@Lk^-T
            # tmp=np.linalg.inv(H0-theta*np.eye(matrixDim))@(Amat-(theta)*np.eye(matrixDim))
            print("Wk\n", Wk)
            print("Lk\n", np.diag(Lk))
            print("p:", p)
            # print('tmp\n',tmp)
            # assert np.allclose(Wk,tmp)

            yk = Lk.T @ p
            print((yk.T @ Wk) @ yk)
            yk = yk / np.linalg.norm(yk)
            # sys.exit()
            # p=p/np.linalg.norm(p) # check for normalization of p

            # Run standard Lanczos algorithm using an improved, transformed initial guess, |p>, and transformed
            #       matrix, Wk
            convgeItr, eVec, ResidList, SSevalList, nextTargeteVec = Run_Lanczos(
                Wk,
                matrixDim,
                microItr,
                np.reshape(yk, (matrixDim, 1)),
                10**-5,
                2,
                PreLanczosBOOL,
                spectraType="lowest",
            )
            mults = mults + convgeItr
            PrecondLanz_matrixVecMultiply.append(mults)

            # Lanczos returns a 'improved' vector, eVec, which is used to define a new initial guess vector |p>
            print("eVec.T@Amat@eVec: ", (eVec.T @ Wk) @ eVec)
            print(np.shape(p))

            p = np.linalg.inv(Lk).T @ eVec
            p = p / np.linalg.norm(p)
            print("Checking if transformed p is normalized: ", np.linalg.norm(p))
            print(np.shape(Lk), np.shape(eVec), np.shape(p))

            # Check for convergence of Preconditioned Lanczos
            theta = ((p.T @ Amat) @ p) / (np.dot(p, p))

            # Get new H0 def, ie QHQ
            if QHQapprox == "updateH0":
                H0 = update_H0def(Amat, p, theta, matrixDim, H0)

            # p=p/np.linalg.norm(p)
            print("Updated estimate of root: ", theta)
            planczos_resid = Amat @ p - theta * p
            norm = np.linalg.norm(planczos_resid)
            PrecondLanz_ResidList.append(norm)
            PrecondLanz_SSevalList.append(theta)
            print("Updated estimate of residual norm: ", norm, TOL, norm < TOL)

            print("new H0 \n", H0)
            if norm < TOL:
                print(
                    "\n ** PRECONDITIONED LANCZOS CONVERGED LOWEST ROOT ON ITERATION: ",
                    i,
                )
                print("** Required ", mults, "matrix-vector multiplies.")
                print("Value of lowest root: ", theta)
                print("Residual information: ", norm, "\n\n")
                # Building the initial guess for the next iteration, if there is one
                convgTheta.append(theta)
                convgVecs.append(np.reshape(p, (matrixDim, 1)))

                # p=np.linalg.inv(Lk).T@nextTargeteVec
                # p=p/np.linalg.norm(p)
                # print('Estimate of next iterations root :', (p.T@Amat)@p)
                final_matmuls.append(PrecondLanz_matrixVecMultiply)
                final_residList.append(PrecondLanz_ResidList)
                final_evals.append(PrecondLanz_SSevalList)
                break

    return final_matmuls, final_residList, final_evals
