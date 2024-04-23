'''
Critical Cost Efficiency Index
==============================
CCEI is defined as supremum of the set of all `e` such that 
the graph generated by the dataset has no cycle under the constraint 
pTqT >= e * pTqV.

Reference:
Afriat, Sidney N. 
"Efficiency estimation of production functions." 
International economic review (1972): 568-598.
'''

import numpy as np
import pulp as pl

from ._utils import (
    generate_graph as _generate_graph, 
    has_cycle as _has_cycle
)

def _warshall_ccei(p: np.ndarray, q: np.ndarray):
    '''
    Floyd-Warshall algorithm for CCEI

    This method is the most efficient way to calculate CCEI.
    Time complexity is O(n^3) where n is the number of vertices.
    '''
    N = q.shape[0]
    PQT = p @ q.T
    EXP = np.diag(PQT).reshape(N, 1) @ np.ones((1, N))
    EXP = np.where(EXP == 0, 1e-7, EXP)
    
    Cs = np.clip(EXP - PQT, 0, None) / EXP
    Es = Cs

    for k in range(len(Cs)):
        for j in range(len(Cs)):
            for i in range(len(Cs)):
                Es[i, j] = max(min(
                    max(Es[i, k],0), max(Es[k, j],0)
                ), Es[i, j])
    
    return 1 - max(np.diag(Es))

def _dichotomy_ccei(p: np.ndarray, q: np.ndarray, tol = 1e-6):
    '''
    Dichotomy algorithm for CCEI
    
    This method is the most standard way to calculate CCEI.
    '''
    f = lambda iccei, p, q: _has_cycle(_generate_graph(p, q, iccei))
    l, r = 0, 1
    while r - l > tol:
        mid = (l + r) / 2
        if f(mid, p, q):
            r = mid
        else:
            l = mid
    return l

def _mtz_ccei(p:np.ndarray, q:np.ndarray, solver = 'PULP_CBC_CMD'):
    '''
    MTZ algorithm for CCEI
    
    Using Mixed Integer Linear Programming (MILP) to calculate CCEI.
    '''
    sl = pl.getSolver(solver, msg = 0)
    edges = list(_generate_graph(p, q).edges())
    T = q.shape[0]
    PQT = p @ q.T
    
    EPSILON = 1/(2 * T)
    ALPHA = np.max(np.diag(PQT)) + 1

    vu = [
        pl.LpVariable(
            "u_{}".format(i + 1), 
            lowBound = 0, upBound = 1, 
        cat=pl.LpContinuous) 
    for i in range(T)]
    
    vU = np.array(
    [
        [
            pl.LpVariable(f"U_{t+1}_{v+1}", cat=pl.LpBinary) 
        for v in range(T)] 
    for t in range(T)])
    
    vc = pl.LpVariable(
            "e_ccei", 
            lowBound = 0, upBound = 1, 
    cat=pl.LpContinuous)
    
    prob = pl.LpProblem("CCEI_Problem", pl.LpMaximize)
    
    for edge in edges:
        t, v = edge
        prob += 2 * vU[t, v] - EPSILON >= vu[t] - vu[v] 
        prob += (vU[t, v] -1) <= vu[t] - vu[v] 
        prob += vc * (PQT[t, t]) - (PQT[t, v]) <= ALPHA * vU[t, v]
        prob += ALPHA * (vU[v, t] - 1) <=  (PQT[t, v] - vc * (PQT[t, t]))
        
    prob += vc
    prob.solve(solver = sl)
    e = pl.value(prob.objective) 
    
    return e

###########################
ccei_warshall = _warshall_ccei
ccei_dichotomy = _dichotomy_ccei
ccei_mtz = _mtz_ccei

__all__ = ['ccei_warshall', 'ccei_dichotomy', 'ccei_mtz']