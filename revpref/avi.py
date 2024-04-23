'''
(AVI - Average Varian Index)
===============================
Various types of Varian indices can be obtained by maximizing some increasing function
of e subject to e-GARP. Here, we focus on the average Varian index (AVI) which maximizes
the average of the vector e.

Reference:
Varian, Hal R. 
"Goodness-of-fit in optimizing models." 
Journal of Econometrics 46, no. 1-2 (1990): 125-140.

Computing the Varian Index is an NP-hard problem.
'''

import numpy as np
import pulp as pl
from ._utils import (
    generate_graph as _generate_graph, 
    has_cycle as _has_cycle
)


def _mtz_avi(p:np.ndarray, q:np.ndarray, solver = 'PULP_CBC_CMD'):
    sl = pl.getSolver(solver, msg = 0)
    edges = list(_generate_graph(p, q).edges())
    T = q.shape[0]
    PQT = p @ q.T
    
    cons_epsilon = 1/(1.01 * T)
    cons_alpha = np.max(np.diag(PQT)) + 1

    var_u = [
        pl.LpVariable(
            "u_{}".format(i + 1), 
            lowBound = 0, upBound = 1, 
        cat=pl.LpContinuous) 
    for i in range(T)]
    
    var_U = [
        [
            pl.LpVariable(f"U_{t+1}_{v+1}", cat=pl.LpBinary) 
        for v in range(T)] 
    for t in range(T)]
    
    var_e = [
        pl.LpVariable(
            "e_{}".format(i+1), 
            lowBound = 0, upBound = 1, 
        cat=pl.LpContinuous) 
    for i in range(T)]
    
    prob = pl.LpProblem("AVI_Problem", pl.LpMaximize)
    
    for edge in edges:
        t, v = edge
        prob += 2*var_U[t][v] - cons_epsilon >= var_u[t] - var_u[v] # IP-1
        prob += (var_U[t][v] -1) <= var_u[t] - var_u[v] # IP-2
        prob += var_e[t] * (PQT[t][t]) - (PQT[t][v]) <= cons_alpha * var_U[t][v] # IP-7
        prob += cons_alpha * (var_U[v][t] - 1) <=  (PQT[t][v] - var_e[t] * (PQT[t][t])) # IP-8
        
    prob += pl.lpDot(var_e, np.ones(T))
    prob.solve(solver = sl)
    e = pl.value(prob.objective) / T
    
    return e

###########################
avi_mtz = _mtz_avi

__all__ = ['avi_mtz']