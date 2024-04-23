import numpy as np
import itertools

def _matrix_mpi_legacy(P:np.ndarray, Q:np.ndarray):
    '''
    The specialized version of MPI computation for the case of cycle length of 2.
    '''
    OBS = P.shape[0]
    P = P.T
    Q = Q.T

    def nchoosek(startnum, endnum, step=1, n=2) -> np.ndarray:
        c = [i for i in itertools.combinations(range(startnum, endnum+1, step), n)]
        return np.array(c)

    permute = lambda nums: np.array(list(itertools.permutations(nums)))

    def ccomputeMPI(P, Q):
        PXMatrix = P.T @ Q
        P1X1 = np.diag(PXMatrix)
        P1X2 = np.diag(np.roll(PXMatrix.T, -1, axis=0).T)
        MPI = np.prod((P1X1 > P1X2))*np.sum(P1X1-P1X2)/np.sum(P1X1)

        return MPI

    x = None
    MPI_Mat = []
    kiringPW = nchoosek(0, OBS-1, step=1, n=2)

    for mm in range(np.size(kiringPW, 0)):
        tempPWPart = permute(list(kiringPW[mm, :]))
        for kk in range(np.size(tempPWPart, 0)):
            minindex = np.argmin(tempPWPart[kk, :])

            tempPWPart[kk, :] = np.roll(tempPWPart[kk, :], -(minindex), axis=0)

        tempPWPart = np.array(list(set([tuple(t) for t in tempPWPart])))

        for kk in range(np.size(tempPWPart, 0)):
            Price_vectorPart = P[:, tempPWPart[kk, :]]
            BundlePart = Q[:, tempPWPart[kk, :]]
            MPI = ccomputeMPI(Price_vectorPart, BundlePart)
            MPI_Mat.append(MPI)


    mpi = np.array(MPI_Mat).reshape(-1)
    a = mpi[mpi > 0]
    if len(a) > 0:
        x = np.mean(a)
    else:
        x = 0

    return x

