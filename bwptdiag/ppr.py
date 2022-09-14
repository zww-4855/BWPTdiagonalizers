import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
import sys
import math

from bwptdiag.InitializeBWPT import *
from bwptdiag.NumericalRoutines import *
from bwptdiag.IOhandling import *

def createModelMatrix(matrixDim, sparsity):
    # matrixDim=10
    # sparsity = 0.01
    Amat = np.zeros((matrixDim, matrixDim))
    for i in range(0, matrixDim):
        Amat[i, i] = i + 1
    Amat = Amat + sparsity * np.random.randint(20, size=(matrixDim, matrixDim))
    # print("Amat: \n",Amat)
    u, v = LA.eig(Amat)
    print("Correct roots: ", np.sort(u))
    return Amat


def computeRHR(fullSpace, H, HDim, order, blockDim):
         overlap=fullSpace[:,:order+1].T @ fullSpace[:,:order+1]
         #print('overlap',overlap)
         s,u=LA.eig(overlap)
         #print('little s:',s)
         sDiag=np.zeros((order+1,order+1))#
         np.fill_diagonal(sDiag,s)
         for i in range(order+1):
             sDiag[i,i]=1.0/np.sqrt(sDiag[i,i])
         #print('sdiag:',sDiag)

         X=u.T @ sDiag
         X=X @ u
         RHR=(fullSpace[:,:order+1].T @ H) @ fullSpace[:,:order+1]
         Hprime=X.T @ RHR
         Hprime=Hprime @ X
         try:
             roots,vecs=LA.eig(Hprime)
             indx=roots.argsort()
             theta=roots[indx]
             vecs=vecs[:,indx] # these are C' ; must convert back to C, ie XC'=C
             transformedVecs=X@vecs # this is the back transform step
             usableRoots=[ elem for elem in theta if elem > 0.15 ]
             usableIndx=[x for x in range(len(theta)) if theta[x] > 0.15]
             #print('Usable roots from <R|H|R>: ',usableRoots)
             #print('Usable vecs from <R|H|R>: ',transformedVecs[:,usableIndx])
         except np.linalg.LinAlgError as err:
             print('inf or NaN inside the code performing delta^(-1/2)Hdelta^(-1/2) diagonalization!!')
             print("Formal error message: ")
             print(err)
            
         return theta, vecs

    ## test orthogonalizing fullspace
    # qq,r=LA.qr(fullSpace[:,:order+1])
    # print('orthogonalized contribuition to |R>:',qq[:,order])
    # print('shape: ',np.shape(fullSpace[:,:blockDim+order+1]),np.shape(H))
    #RHR = (fullSpace.T @ H) @ fullSpace  # (qq.T @ H) @ qq
#    print("RHR: \n", RHR)
#    print("RHR size: ", RHR.size)
#    roots, vecs = LA.eig(RHR)
#    print("Roots from <R|H|R> via QR:", np.sort(roots))
#    indx = roots.argsort()
#    theta = roots[indx]
#    vecs = vecs[:, indx]
#    print("C_i vecs: ", vecs)
#    usableRoots = [elem for elem in theta if elem > 0.15]
#    usableIndx = [x for x in range(len(theta)) if theta[x] > 0.15]
#    # print('C_i vecs: ',vecs[:,usableIndx])
#    return roots, vecs  # usableRoots,vecs[:,usableIndx]


def extendPSpace(T0, V, p, order):
    Vp = np.matmul(V, p)
    T0V = np.matmul(T0, Vp)
    tmp = T0V  # @p
    print("inside extendPSpace order: ", order)
    for i in range(1, order+1 ):
        tmp = np.matmul(V, tmp)
        tmp = np.matmul(T0, tmp)

    return tmp


def calcContribs(H, fullSpace, pDim):
    p = fullSpace[:, 0]
    corrVec = fullSpace[:, 1]
    E0 = (p.T @ H) @ p

    corrVec1 = np.reshape(corrVec, (pDim,))
    print("shape of p: ", np.shape(p))
    print("shape of corrVec:", np.shape(corrVec1))
    print("Correction vec inside calcContribs: ", corrVec1)

    pHphi = (p.T @ H) @ corrVec1
    print("<p|h|phi>:", pHphi)
    phiHp = (corrVec1.T @ H) @ p
    print("<phi|H|p>:", phiHp)

    secondOrder = E0 + pHphi + phiHp
    phiHphi = (corrVec1.T @ H) @ corrVec1
    print("<phi|H|phi>:", phiHphi)
    thirdOrder = secondOrder + phiHphi
    print("Breaking down contributions.....")
    print("zeroth order E0:", E0)
    print("second order:: E0+<p|h|phi>+<phi|H|p>", secondOrder)
    print("full third order:: E0+<p|h|phi>+<phi|H|p>+<phi|H|phi>", thirdOrder)


# def createT0_projQ(q,qDim,H0Dim,eps,H0): #qDim>0
#     print('q',q,eps,qDim)
#     projQ=q@q.T
#     sumMatrix=(1.00000000/eps)*projQ
#     print('newT0: ',sumMatrix)

#     return sumMatrix


def defineAllVariables(
    Amat, matrixDim, blockDim, H0def
):  # defines single vector p,q,eps,H0,V from a given matrix and its dimension
    p = np.zeros((matrixDim, blockDim))
    q = np.zeros((matrixDim, matrixDim - blockDim))
    print(np.shape(p), np.shape(q))
    diagMat = np.diag(Amat)
    print("diagonal sorted: ", np.sort(diagMat))

    lowestElements = np.argsort(diagMat)
    pIndx = []
    print("lowestElements: ", lowestElements)
    for i in range(blockDim):
        p[lowestElements[i], i] = 1.0
        pIndx.append(lowestElements[i])

    counter = 0
    for i in range(len(lowestElements)):
        if i in pIndx:  # If i coincides with index that is included in |p>, skip it
            continue
        else:
            q[i, counter] = 1.0
            counter = counter + 1

    print("done seeding q")
    projP = p @ p.T  # np.outer(p,p)
    projQ = q @ q.T
    testI = projP + projQ
    realI = np.identity(matrixDim)
    print("projP+projQ == I ;allclose: ", np.allclose(testI, realI))

    # sys.exit()

    # print('p:',p)
    # print('q:',q)
    # H0=np.diag(np.diag(Amat))#np.matmul(np.matmul(projP,Amat),projP)
    if H0def == "PHP":
        H0 = (projP @ Amat) @ projP
        H0 = np.diag(np.diag(H0))

        V = Amat - H0
        # print('V \n',V)
        # print('original Guess:',diagMat[0])
        H = Amat
        print("Check H0+V == H : ", np.allclose(H, H0 + V))
        print("H0: ", H0)
    elif H0def == "diag":
        H0 = np.diag(np.diag(Amat))
        V = Amat - H0

    elif H0def == "diag_offDiag":
        H0 = np.diag(Amat)
        offDiag1 = (projP @ Amat) @ projQ
        offDiag2 = (projQ @ Amat) @ projP
        H0 = H0 + offDiag1 + offDiag2
        V = Amat - H0

    guessEps = []
    for x in range(blockDim):
        guessEps.append(diagMat[lowestElements[x]])  # -0.001)

    eps = guessEps
    print("init Guesses:", eps)
    return p, q, H0, V, eps


def createT0_projQ(projQ, qDim, H0Dim, eps, H0, order, H0def):  # qDim>0
    # print('q',q,eps,qDim)
    # projQ=q@q.T
    if H0def == "PHP":
        QH0Q = (projQ @ H0) @ projQ
        testZeros = np.zeros((H0Dim, H0Dim))
        print("Check QH0Q == 0 for this choice of H0: ", np.allclose(QH0Q, testZeros))
        # eps[0]=eps[0]+0.1
        epsMat = np.zeros((H0Dim, H0Dim))
        denom = np.zeros((H0Dim, H0Dim))
        np.fill_diagonal(epsMat, eps)
        denom = epsMat  # -H0
        print("eps", denom)
        # denom=inv(denom)
        for zzz in range(H0Dim):
            xxx = epsMat[zzz, zzz]  # -diagAmat[zzz]
            if abs(xxx) < 0.0001:
                xxx = math.copysign(0.0001, xxx)
            denom[zzz, zzz] = 1.0000000 / xxx
        print("denom eps 1/eps", denom)
    elif H0def == "diag" or H0def == "diag_offDiag":
        QH0Q = (projQ @ H0) @ projQ
        testZeros = np.zeros((H0Dim, H0Dim))
        print(
            "Check QH0Q != 0 for this choice of H0==diag(H): ",
            np.allclose(QH0Q, testZeros),
        )
        # eps[0]=eps[0]+0.1
        epsMat = np.zeros((H0Dim, H0Dim))
        denom = np.zeros((H0Dim, H0Dim))
        np.fill_diagonal(epsMat, eps)
        denom = epsMat - H0
        print("eps", denom)
        # denom=inv(denom)
        print("H0dim", H0.shape, H0.size)
        print("epsMatdim", epsMat.shape, epsMat.size)
        for zzz in range(H0Dim):
            xxx = denom[zzz, zzz]
            if abs(xxx) < 0.0001:
                xxx = math.copysign(0.0001, xxx)
            denom[zzz, zzz] = xxx
        print("denom eps -diag(H0)", denom)
        denom = inv(denom)

    #     sumMatrix=np.zeros((H0Dim,H0Dim))
    #     for i in range(qDim):
    #         Q=np.outer(q[:,i],q[:,i])
    #         sumMatrix=sumMatrix+Q

    #     sumMatrix=denom@sumMatrix
    sumMatrix = denom @ projQ
    print("total Resolvent Q/eps", sumMatrix)
    return projQ, sumMatrix


def runRedefiningH0_lowOrder(
    H, Hdim, rootTarget, maxorder, blockDim, maxiter, H0def, TOL
):
    # H0def has two defn:
    # PHP = just the eigenvalues of HP|p>, zeros elsewhere. Convenient choice bc the
    #       (eps - QH0Q) denominator of the the R0 goes to 0. Equivalent to moment expansion
    # diag = the diagonal of the original H; PHP +P'HP'. Must compensate for resolvent
    # diag_offDiag = the diagonal of original H plus the whole row/column corresponding to the
    #                original guess |p> in question. Must compensate for resolvent.
    # rootTarget is which root to target; ie lowest root is rootTarget=1
    unconvergedList = np.arange(0, blockDim)
    for xx in range(1):
        # print('Searching for root number: ',xx,rootNumber[xx])
        #p, q, H0, V, eps = defineAllVariables(H, Hdim, blockDim, H0def)

        p, eps, lowestElements = Get_p(H, Hdim, blockDim, True)
        H0, V = Get_H0(H, p, Hdim, blockDim, H0def)



        # print('H0:',H0)
        fullSpace = np.zeros((Hdim, Hdim))
        testfullSpace = np.zeros((Hdim, Hdim))
        # print('shape of FS and p',np.shape(fullSpace[:,0]),np.shape(p))
        fullSpace[:, np.arange(0, blockDim)] = p
        order = 0
        #qDim = Hdim - blockDim
        Hbkup = H
        H0bkup = H0
        #qBKUP = q
        # print('initial guess H0: ',eps)
        finalEps = []
        tmpRoots = np.zeros((Hdim, Hdim))
        #projQ = q @ q.T
        projQ=np.eye(Hdim)-np.outer(p,p)
        residuals = []
        convergedIndx = []
        # Macro-iteration of the redefineH0 algo
        for i in range(maxiter):
            print("*********************************")
            print("*********************************")
            print("***** Iteration: ", i)
            # Micro-iterations of redefineH0 algo
            # Purpose::: builds 'maxorder' correction vectors, then appends it to 'fullSpace'
            #            The building of correction vectors applies to either one 'rootTarget'
            #            of if 'rootTarget==1', then the **BLOCK** algorithm is performed
            for order in range(0, maxorder):
                print(
                    "Inside inner loop with current order: ",
                    order + 1,
                    "out of a max. order: ",
                    maxorder + 1,
                )
                if (
                    rootTarget != -1
                ):  ## rootTarget intends to target particular root, ie 0th, 1st, 2nd, etc...
                    _, T0 = createT0_projQ(
                        projQ, qDim, Hdim, eps[rootTarget], H0, order, H0def
                    )
                    pertVec = extendPSpace(
                        T0, H, p[:, rootTarget], order
                    )  # first order correction
                    # order=order+1
                    #                 print('pertVec:',pertVec)
                    print('norm of pertVec: ', np.linalg.norm(pertVec))
                    pertVec=pertVec/np.linalg.norm(pertVec)
                    pertVec = np.reshape(pertVec, (Hdim, 1))
                    #                 print('shape of pertVec*****:',np.shape(pertVec))
                    #                 print('shape of fullSpace: ',np.shape(fullSpace))
                    var = blockDim + order  # i*blockDim+order
                    print("var storing fS locaiton is: ", var)
                    fullSpace[:, [var]] = pertVec

                else:  ## if rootTarget==-1, then run **BLOCK** Algo
                    blockIndx = 0
                    blockVar = blockDim - len(convergedIndx)
                    for aa in range(blockDim):
                        if (
                            aa in convergedIndx
                        ):  # dont expect this vector as its converged already
                            continue
                        else:
                            _, T0 = createT0_projQ(
                                projQ, qDim, Hdim, eps[blockIndx], H0, order, H0def
                            )
                            pertVec = extendPSpace(
                                T0, H, p[:, blockIndx], order
                            )  # first order correction
                            pertVec = np.reshape(pertVec, (Hdim, 1))
                            var = blockDim + order * blockVar + blockIndx
                            blockIndx = blockIndx + 1
                            print("var storing fS locaiton is: ", var)
                            fullSpace[
                                :, [blockDim + order * blockVar + blockIndx]
                            ] = pertVec

            print("order after exiting inner loop: ", order)
            #             print('Current fullSpace: B4 orthogonalization',fullSpace[:,:order+1])
            # Orthogonalize all vectors in 'fullSpace'
            #qq, r = LA.qr(fullSpace[:, : var + 1])

            #             print('shape of QR decomp output for orthogonal FS: ',np.shape(qq))
            #             #fullSpace[:,:order+1]=qq

            #fullSpaceTMP = qq
            ##
            #             print('Current fullSpace: after orthogonalization',fullSpaceTMP[:,:order+1])
            #print('fullSpace:', fullSpace[:,:order+3])
            #sys.exit()
            ritzRoots, ritzVecs = computeRHR(fullSpace, H, Hdim, order+1, blockDim)
            #            print('ritzVecs:',ritzVecs)
            print("ritzRoots:", np.sort(ritzRoots))
            eVec=fullSpace[:,:var+1]@ritzVecs[:,0] #qq[:,:var+1]@ritzVecs[:,0]
            print('test of eVec:', eVec.T @ (H@eVec))
            # calcContribs(H,fullSpace,Hdim)

            fatPhi = np.zeros((Hdim, blockDim))
            eps = np.sort(ritzRoots)[:blockDim]  # [rootTarget-1]#-0.01
            finalEps.append(eps)

            for zz in range(blockDim):
                fatPhi[:, zz] = (
                    fullSpace[:, : var + 1] @ ritzVecs[:, zz]
                )  # approx to the lowest energy eigenvector
                fullSpace[:, zz] = fatPhi[:, zz]

            p = fatPhi
            projFatPhi = np.zeros((Hdim, Hdim))
            print("Check overlap <fatPhi|fatPhi>:", fatPhi.T @ fatPhi)
            projFatPhi = fatPhi @ fatPhi.T  # np.outer(fatPhi,fatPhi)#p@p.T#

            # Redefine H0 w.r.t the original H0 define prior to any iterations
            #H0 = np.zeros((Hdim, Hdim))

            #H0 = eVec@(H0bkup@eVec)#(projFatPhi @ H0bkup) @ projFatPhi
            V = H - H0

            # Test that after I-projFatPhi==projQ, then projQ+projFatPhi == I
            testI = projFatPhi + projQ
            realI = np.identity(Hdim)
            # print('Checking allclose for new projP+projQ=I : ',np.allclose(testI,realI))

            testProjQ = realI - projFatPhi
            testI = projFatPhi + testProjQ
            #            print('Checking allclose for redefined projP&&projQ; aka new projP+projQ==I where \n projP=fatPhi @ fatPhi.T \n new projQ=I-projP \n ',np.allclose(testI,realI))
            projQ = realI - projFatPhi

            #             combinedV=(projFatPhi@H)@projQ
            #             combinedV=combinedV+(projQ@H)@projFatPhi
            #             combinedV=combinedV+(projQ@H)@projQ
            #             print('Check V=H-H0 ==? V=PHQ+QHP+QHQ',np.allclose(V,combinedV))
            #            print('V=H-H0:',V)
            #            print('V=PHQ+QHP+QHQ:',combinedV)
            #            combinedH=combinedV+(projFatPhi@H)@projFatPhi
            #            print('Check full H ==? newly defined PHP+PHQ+QHP+QHQ',np.allclose(H,combinedH))
            #            print('Redefined H0: \n',H0)

            # Check for convergence criteria
            if rootTarget != -1:  # we are targeting a specific root
                transformedVec = fatPhi[
                    :, rootTarget
                ]  # fullSpace[:,:order+1]@ritzVecs[:,0]
                np.fill_diagonal(tmpRoots, np.sort(ritzRoots)[rootTarget])
                residualVec = (H - np.sort(ritzRoots)[0]*np.eye(Hdim)) @ fatPhi[:,0]  # [:,rootTarget-1]
                normResid = LA.norm(residualVec)
                print("Norm of rootTarget residual inside sinlge vec algo: ", normResid)
                print("tolerance is: ", TOL, normResid < TOL)
                residuals.append(normResid)
                #import sys
                #sys.exit()
                if normResid < TOL:
                    print(
                        "\n\n\n Converged in micro-iteration: ",
                        (order + 1) * (i + 1),
                        " macro-iteration:",
                        i + 1,
                    )
                    print(
                        "Converged root ",
                        rootTarget,
                        " :",
                        np.sort(ritzRoots)[rootTarget],
                    )
                    return finalEps, residuals, convergedIndx

            else:  # we are in a block scheme
                tmpResid = []
                for zzz in range(blockDim):
                    if zzz in convergedIndx:  # don't incorporate converged roots
                        continue
                    transformedVec = fatPhi[
                        :, zzz
                    ]  # fullSpace[:,:order+1]@ritzVecs[:,0]
                    np.fill_diagonal(tmpRoots, np.sort(ritzRoots)[zzz])
                    residualVec = (H - tmpRoots) @ transformedVec  # [:,rootTarget-1]
                    normResid = LA.norm(residualVec)
                    print("Norm of residual: ", normResid)
                    tmpResid.append(normResid)
                    if normResid < TOL:
                        print(
                            "\n\n\n Converged on iteration: ",
                            (i + 1) * order,
                            rootTarget,
                            zzz,
                        )
                        print(
                            "Converged root ",
                            rootTarget,
                            " :\n\n\n",
                            np.sort(ritzRoots)[zzz],
                        )
                        convergedIndx.append(zzz)
                        fullSpace[:, zzz] = fatPhi[:, zzz]
                residuals.append(tmpResid)
                if len(convergedIndx) == blockDim:
                    return finalEps, residuals, convergedIndx

            order = 0
            # if rootTarget!=-1:
            #    if len(convergedIndx)==1:
            #        return finalEps,residuals,convergedIndx

    return finalEps, residuals, convergedIndx
