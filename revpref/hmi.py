'''
HMI - Houtman-Maks Index
==================
The Houtman-Maks index gives the relative size of 
the largest subset of observations that is still consistent with GARP.


Reference:
Houtman, Martijn, and Julian Maks. 
"Determining all maximal data subsets consistent with revealed preference." 
Kwantitatieve methoden 19, no. 1 (1985): 89-104.
==================
Computing the Houtman-Maks index is an NP-hard problem.
'''

import numpy as np
import pulp as pl
import networkx as nx
from scipy.optimize import milp, LinearConstraint, Bounds

from ._utils import (
    generate_graph as _generate_graph, 
    has_cycle as _has_cycle
)

def _mtz_hmi(p:np.ndarray, q:np.ndarray, solver = 'PULP_CBC_CMD'):
    sl = pl.getSolver(solver, msg = 0)
    edges = list(_generate_graph(p, q).edges())
    T = q.shape[0]
    PQT = p @ q.T
    
    EXP = np.diag(PQT).reshape(T, 1) @ np.ones((1, T))
    Cs = np.clip(PQT - EXP, 0, None)
    Cmin = np.min(Cs[np.nonzero(Cs)])
    Dmin = np.min(PQT[np.nonzero(PQT)])
    
    DELTA = min(Cmin, Dmin) / 2
    EPSILON = 1/(2 * T)
    ALPHA = np.max(PQT) + 1
    
    vu = [
        pl.LpVariable(
            "u_{}".format(i + 1), 
            lowBound = 0, upBound = 1, 
        cat=pl.LpContinuous) 
    for i in range(T)]
    
    vA = [
        pl.LpVariable(
            "A_{}".format(i + 1), 
        cat=pl.LpBinary) 
    for i in range(T)]
    
    vU = [
        [
            pl.LpVariable(f"U_{t+1}_{v+1}", cat=pl.LpBinary) 
        for v in range(T)] 
    for t in range(T)]
    
    prob = pl.LpProblem("HMI_Problem", pl.LpMaximize)
    for edge in edges:
        t, v = edge
        prob += 2 * vU[t][v] - EPSILON >= vu[t] - vu[v] # IP-1
        prob += (vU[t][v] -1) <= vu[t] - vu[v] # IP-2
        
        prob += -DELTA + ALPHA * (vU[t][v] + 1 - vA[t]) >= (PQT[t][t] - PQT[t][v]) 
        prob += ALPHA * (vU[v][t] + vA[t] - 2) <= PQT[t][v] - PQT[t][t]
    
    prob += pl.lpSum(vA)
    prob.solve(sl)
    e = pl.value(prob.objective) / T
    
    return e

def _gross_hmi(p:np.ndarray, q:np.ndarray):
    '''
    Gross-Kaiser Algorithm (1994)
    
    '''
    def AdjMat(p:np.ndarray, q:np.ndarray):
        T=np.size(p,0)
        DRP=np.zeros((T,T))
        P0=np.zeros((T,T))

        for i in range(T):
            for j in range(T):
                if (p[i,:]*q[i,:]).sum()>= (p[i,:]*q[j,:]).sum():
                    DRP[i,j]=1
                if (p[i,:]*q[i,:]).sum()> (p[i,:]*q[j,:]).sum():
                    P0[i,j]=1

        A=np.zeros((T,T))
        for i in range(T):
            for j in range(T):
                if DRP[i,j]==1 and P0[j,i]==1:
                    A[i,j]=1
                if P0[i,j]==1 and DRP[j,i]==1:
                    A[i,j]=1

        return A


    def NonAdjRem(A):
        degr=np.sum(A.T,axis=1).T
        degrVec=np.array(np.where(degr>0)).reshape(-1,)

        RemList=[]

        if len(degrVec)==0:
            return RemList  
        else:
            degrMax=np.array(np.where(degr==np.max(degr))).reshape(-1,)
            ind=0

            for i in degrMax:
                Ai=np.array(np.where(A[i,:]==1)).reshape(-1,)
                if degr[i]>np.min(degr[Ai]):
                    ind=ind+1
                    if ind==1:
                        RemList=[i]           
        return RemList  

    def AdjRem( A ):
        degr=np.sum(A.T,axis=1).T
        degrVec=np.array(np.where(degr>0)).reshape(-1,)
        RemList=[]

        if len(degrVec)==0:
            return RemList  
        else:
            degrMax=np.array(np.where(degr==np.max(degr))).reshape(-1,)
            ind=0

            for i in degrMax:
                Ai=np.array(np.where(A[i,:]==1)).reshape(-1,)
                Ai1=Ai[np.array(np.where(degr[Ai]==1)).reshape(-1,)]

                for h in degrMax:
                    Ah=np.array(np.where(A[h,:]==1)).reshape(-1,)
                    Ah1=Ah[np.array(np.where(degr[Ah]==1)).reshape(-1,)]

                    if h in Ai:
                        ind=ind+1
                        if ind==1:
                            if len(Ai1)>0:
                                RemList.append(i)
                            elif len(Ah1)>0:
                                RemList.append(h)
                            elif len(Ai1)==0 and len(Ah1)==0:
                                RemList.append(i)        
        return RemList       
    
    T=np.size(p,0)
    ListOfIndices = np.arange(0,T)
    r=0
    
    while r==0:
        pTemp = p[ListOfIndices,:]
        qTemp = q[ListOfIndices,:]

        edgesTemp = AdjMat(pTemp,qTemp)
        remove = NonAdjRem(edgesTemp)
        
        r =(len(remove)==0)

        if r==0:
            ListOfIndices=np.delete(ListOfIndices,remove)

    r=0

    while r==0:
        pTemp = p[ListOfIndices,:]
        qTemp = q[ListOfIndices,:]

        edgesTemp = AdjMat(pTemp,qTemp)
        remove = AdjRem(edgesTemp)
        r =(len(remove)==0)

        if r==0:
            ListOfIndices=np.delete(ListOfIndices,remove)
    
    return len(ListOfIndices)/T

def _optimize_hmi(p:np.ndarray, q:np.ndarray):
    G = _generate_graph(p, q)
    N = p.shape[0]
    max_depth = 3
    min_remove = 0 
    if not _has_cycle(G):
        return 1
    while _has_cycle(G):
        cycles = list(nx.simple_cycles(G, max_depth))
        lc = len(cycles)
        lp_remove = np.ones(N)
        lp_matrix = np.zeros((lc, N), dtype = np.int8)
        
        lp_lower = np.ones(lc, dtype = np.int8)
        
        lp_bounds = Bounds(0, 1)
        for c, cycle in enumerate(cycles):
            for i in range(N):
                lp_matrix[c][i] = 1 if i in cycle else 0
        lp_cons = LinearConstraint(lp_matrix, lp_lower)
        lp_hmi = milp(lp_remove, constraints = lp_cons, bounds = lp_bounds)
        if lp_hmi.success:
            g_check = G.copy()
            for n in range(N):
                if lp_hmi.x[n] == 1:
                    g_check.remove_node(n)
            if _has_cycle(g_check):
                max_depth += 1
            else:
                min_remove = lp_hmi.fun
                return 1 - min_remove / N
        else:
            raise RuntimeError("MILP Failed!")
        

######################

hmi_mtz = _mtz_hmi
hmi_gross = _gross_hmi
hmi_optimize = _optimize_hmi

__all__ = ['hmi_mtz', 'hmi_gross']