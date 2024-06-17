'''
Measurement Error
==============================
These methods are intuitive, but might also be useful for statistical
tests that incorporate measurement error.

AEE(Average Expenditure Error) 
Relax the GARP condition: p^tq^t - r^t >= p^tq^v
and find the minimum sum of r^t that satisfies the condition.

Reference:
Dean, Mark, and Daniel Martin. 
"Measuring rationality with the minimum cost of revealed preference violations." 
Review of Economics and Statistics 98, no. 3 (2016): 524-534.
'''

import numpy as np
import pulp as pl

from ._utils import (
    generate_graph as _generate_graph, 
    has_cycle as _has_cycle,
    find_solver as _find_solver
)

def _mtz_aee(p:np.ndarray, q:np.ndarray, solver = 'PULP_CBC_CMD'):
    '''
    MTZ algorithm for AEE
    
    Using Mixed Integer Linear Programming (MILP) to calculate AEE.
    '''
    sl = _find_solver(solver, msg = 0)
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
    
    var_r = np.array(
    [
        pl.LpVariable(
            "r_{}".format(i+1), 
            lowBound = 0, upBound = np.max(PQT), 
        cat=pl.LpContinuous) 
    for i in range(T)])
    
    prob = pl.LpProblem("AEE_Problem", pl.LpMinimize)
    
    for edge in edges:
        t, v = edge
        prob += 2 * vU[t, v] - EPSILON >= vu[t] - vu[v] 
        prob += (vU[t, v] -1) <= vu[t] - vu[v] 
        prob += (PQT[t, t] - PQT[t, v] - var_r[t]) <= ALPHA * vU[t, v]
        prob += ALPHA * (vU[v, t] - 1) <=  (PQT[t, v] - PQT[t, t] + var_r[t])
        
    prob += pl.lpDot(var_r, np.ones(T))
    prob.solve(solver = sl)
    e = pl.value(prob.objective) / T
    
    return e

def _mtz_ape(p:np.ndarray, q:np.ndarray, solver = 'PULP_CBC_CMD'):
    raise NotImplementedError("APE not implemented")