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
    find_solver as _find_solver,
    RevprefError
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

    vu = np.array(
    [
        pl.LpVariable(
            "u_{}".format(i + 1), 
            lowBound = 0, upBound = 1, 
        cat=pl.LpContinuous) 
    for i in range(T)])
    
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
    '''
    MTZ algorithm for APE
    '''
    sl = _find_solver(solver, msg = 0)
    edges = list(_generate_graph(p, q).edges())
    T = q.shape[0]
    K = p.shape[1]
    PQT = p @ q.T
    
    EPSILON = 1/(2 * T)
    ALPHA = np.max(np.diag(PQT)) * 1.5
    
    vu = np.array(
    [
        pl.LpVariable(
            "u_{}".format(i + 1), 
            lowBound = 0, upBound = 1, 
        cat=pl.LpContinuous) 
    for i in range(T)])
    
    vU = np.array(
    [
        [
            pl.LpVariable(f"U_{t+1}_{v+1}", cat=pl.LpBinary) 
        for v in range(T)] 
    for t in range(T)])
    
    vp = np.array(
    [
        [
            pl.LpVariable(f"p_{t+1}_{i+1}", cat=pl.LpContinuous) 
        for i in range(K)] 
    for t in range(T)])
    
    vd = np.array(
    [
        [
            pl.LpVariable(f"delta_{t+1}_{i+1}", cat=pl.LpContinuous, lowBound = 0) 
        for i in range(K)] 
    for t in range(T)])
    
    prob = pl.LpProblem("APE_Problem", pl.LpMinimize)
    
    for edge in edges:
        t, v = edge
        prob += 2 * vU[t, v] - EPSILON >= vu[t] - vu[v] 
        prob += (vU[t, v] -1) <= vu[t] - vu[v] 
        IP15 = pl.lpDot([vp[t, i] for i in range(K)], q[t]) - pl.lpDot([vp[t, i] for i in range(K)], q[v])
        prob += IP15 <= ALPHA * vU[t, v]
        IP16 = pl.lpDot([vp[t, i] for i in range(K)], q[v]) - pl.lpDot([vp[t, i] for i in range(K)], q[t])
        prob += ALPHA * (vU[v, t] - 1) <= IP16
        for i in range(K):
            prob += vp[t, i] - p[t, i] <= vd[t, i]
            prob += p[t, i] - vp[t, i] <= vd[t, i]
        prob += pl.lpSum([vp[t, i] for i in range(K)]) == np.sum(p[t])

    prob += pl.lpSum([pl.lpSum([vd[t, i] for i in range(K)])/np.sum(p[t]) for t in range(T)])
    prob.solve(solver = sl)
    if prob.status != pl.LpStatusOptimal:
        raise RevprefError(f"Optimization failed with status: {pl.LpStatus[prob.status]}")
    e = pl.value(prob.objective) / T
    return e


def _mtz_aqe(p:np.ndarray, q:np.ndarray, solver = 'PULP_CBC_CMD'):
    '''
    MTZ algorithm for APE
    '''
    sl = _find_solver(solver, msg = 0)
    edges = list(_generate_graph(p, q).edges())
    T = q.shape[0]
    K = q.shape[1]
    PQT = p @ q.T
    
    EPSILON = 1/(2 * T)
    ALPHA = np.max(np.diag(PQT)) * 1.5
    
    vu = np.array(
    [
        pl.LpVariable(
            "u_{}".format(i + 1), 
            lowBound = 0, upBound = 1, 
        cat=pl.LpContinuous) 
    for i in range(T)])
    
    vU = np.array(
    [
        [
            pl.LpVariable(f"U_{t+1}_{v+1}", cat=pl.LpBinary) 
        for v in range(T)] 
    for t in range(T)])
    
    vq = np.array(
    [
        [
            pl.LpVariable(f"q_{t+1}_{i+1}", cat=pl.LpContinuous) 
        for i in range(K)] 
    for t in range(T)])
    
    vd = np.array(
    [
        [
            pl.LpVariable(f"delta_{t+1}_{i+1}", cat=pl.LpContinuous, lowBound = 0) 
        for i in range(K)] 
    for t in range(T)])
    
    prob = pl.LpProblem("AQE_Problem", pl.LpMinimize)
    
    for edge in edges:
        t, v = edge
        prob += 2 * vU[t, v] - EPSILON >= vu[t] - vu[v] 
        prob += (vU[t, v] -1) <= vu[t] - vu[v] 
        IP15 = pl.lpDot(p[t], [vq[t, i] for i in range(K)]) - pl.lpDot(p[t], [vq[v, i] for i in range(K)])
        prob += IP15 <= ALPHA * vU[t, v]
        IP16 = pl.lpDot(p[t], [vq[v, i] for i in range(K)]) - pl.lpDot(p[t], [vq[t, i] for i in range(K)])
        prob += ALPHA * (vU[v, t] - 1) <= IP16
        for i in range(K):
            prob += vq[t, i] - q[t, i] <= vd[t, i]
            prob += q[t, i] - vq[t, i] <= vd[t, i]
        prob += pl.lpSum([vq[t, i] for i in range(K)]) == np.sum(q[t])

    prob += pl.lpSum([pl.lpSum([vd[t, i] for i in range(K)])/np.sum(q[t]) for t in range(T)])
    prob.solve(solver = sl)
    if prob.status != pl.LpStatusOptimal:
        raise RevprefError(f"Optimization failed with status: {pl.LpStatus[prob.status]}")
    e = pl.value(prob.objective) / T
    return e
