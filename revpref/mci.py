'''
Minimum Cost Index
==================
MCI is defined as the minimum cost of removing all cycles from the graph.

Reference:
Dean, Mark, and Daniel Martin. 
"Measuring rationality with the minimum cost of revealed preference violations." 
Review of Economics and Statistics 98, no. 3 (2016): 524-534.

Computing MCI is an NP-hard problem.
MILP is used to solve the problem.
'''

import pulp as pl
import numpy as np
import networkx as nx
from scipy.optimize import milp, LinearConstraint, Bounds

from ._utils import (
    generate_graph as _generate_graph, 
    has_cycle as _has_cycle
)

def _check_if_in_cycle(edge, cycle: list):
    try:
        return cycle[(cycle.index(edge[0]) + 1) % len(cycle)] == edge[1]
    except:
        return 0

def _load_matrix(cycles, matrix, Edges):
    for cycle_c in range(len(cycles)):
        for edge_c in range(len(Edges)):
            matrix[cycle_c][edge_c] = _check_if_in_cycle(Edges[edge_c], cycles[cycle_c])
            
def _optimize_mci(p:np.ndarray, q:np.ndarray):
    G = _generate_graph(p, q)
    max_depth = 3
    min_cost = 0
    sum_cost = np.diag(p @ q.T).sum()
    if not _has_cycle(G):
        return 0
    while _has_cycle(G):
        cycles = list(nx.simple_cycles(G, max_depth))
        lc = len(cycles)
        le = len(G.edges())
        lp_weight = np.zeros(le)
        lp_matrix = np.zeros((lc, le), dtype = np.int32)
        P_Edges = np.zeros((le, 2))
        Edges = list(G.edges().data('weight'))
        _load_matrix(cycles, lp_matrix, Edges)
        for c, edge in enumerate(Edges):
            P_Edges[c][0], P_Edges[c][1], lp_weight[c] = edge[0], edge[1], edge[2]
            
        lp_lower = np.ones(lc, dtype = np.int32)
        lp_cons = LinearConstraint(lp_matrix, lp_lower)
        lp_bounds = Bounds(0, 1)
        mci_lp = milp(lp_weight, constraints = lp_cons, bounds = lp_bounds)
        
        if mci_lp.success:
            g_check = G.copy()
            for c, edge in enumerate(Edges):
                if mci_lp.x[c] == 1:
                    g_check.remove_edge(edge[0], edge[1])
            if _has_cycle(g_check):
                max_depth += 1
            else:
                min_cost = mci_lp.fun
                return min_cost / sum_cost
        else:
            raise RuntimeError("MILP Failed!")
    pass

def _mtz_mci(p:np.ndarray, q:np.ndarray, solver = 'PULP_CBC_CMD'):
    sl = pl.getSolver(solver, msg = 0)
    edges = list(_generate_graph(p, q).edges())
    T = q.shape[0]
    PQT = p @ q.T
    
    EXP = np.diag(PQT).reshape(T, 1) @ np.ones((1, T))
    Cs = np.clip(PQT - EXP, 0, None)
    Cmin = np.min(Cs[np.nonzero(Cs)])
    Dmin = np.min(PQT[np.nonzero(PQT)])
    
    cons_delta = min(Cmin, Dmin) / 2
    cons_epsilon = 1/(1.5 * T)
    cons_alpha = np.max(PQT) * 2
    cons_beta = np.max(PQT) * 2
    
    sum_cost = np.diag(PQT).sum()
    
    vu = [
        pl.LpVariable(
            "u_{}".format(i + 1), 
            lowBound = 0, upBound = 1, 
        cat=pl.LpContinuous) 
    for i in range(T)]
    
    vU = [
        [
            pl.LpVariable(f"U_{t+1}_{v+1}", cat=pl.LpBinary) 
        for v in range(T)] 
    for t in range(T)]
    
    vB = [
        [
            pl.LpVariable(f"B_{t+1}_{v+1}", cat=pl.LpBinary) 
        for v in range(T)]
    for t in range(T)]
    
    weight = [
        [
            PQT[t][t] - PQT[t][v]
        for v in range(T)]
    for t in range(T)]
    
    vB2 = [vB[edge[0]][edge[1]] for edge in edges]
    vw = [weight[edge[0]][edge[1]] for edge in edges]
    
    prob = pl.LpProblem("MCI_Problem", pl.LpMinimize)
    
    for edge in edges:
        t, v = edge
        prob += 2 * vU[t][v] - cons_epsilon >= vu[t] - vu[v] # IP-1
        prob += (vU[t][v] -1) <= vu[t] - vu[v] # IP-2
        
        prob += -cons_delta + cons_alpha * (vU[t][v] +  vB[t][v]) >= PQT[t][t] - PQT[t][v]
        prob += cons_alpha * (vU[v][t] - 1 - vB[t][v]) <= PQT[t][v] - PQT[t][t]
        prob += cons_beta * (vB[t][v] - 1) <=  PQT[t][t] - PQT[t][v]
    
    prob += pl.lpDot(vB2, vw)
    prob.solve(solver = sl)
    e = pl.value(prob.objective) / sum_cost
    
    return e
    
def _milp_mci(p:np.ndarray, q:np.ndarray, solver = 'PULP_CBC_CMD'):
    raise NotImplementedError("MILP not implemented")

####################################
mci_mtz = _mtz_mci
mci_milp = _milp_mci
mci_optimize = _optimize_mci

__all__ = ['mci_mtz', 'mci_milp', 'mci_optimize']