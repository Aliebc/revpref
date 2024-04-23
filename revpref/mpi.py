'''
Money Pump Index
================
This module contains the functions to compute the Money Pump Index(MPI) of a given dataset.
The MPI is a measure of the potential for money pump in a given dataset. 
It assumes that there is a arbitrary agent who can exploit the consumer by buying and selling goods in a cycle.

Reference:
Echenique, Federico, Sangmok Lee, and Matthew Shum. 
"The money pump as a measure of revealed preference violations." 
Journal of Political Economy 119, no. 6 (2011): 1201-1223.
'''

import numpy as np
import networkx as nx

from ._utils import (
    generate_graph as _generate_graph, 
    has_cycle as _has_cycle
)

def _cycle_mpi(p:np.ndarray, q:np.ndarray, max_depth = 2):
    '''
    Compute the Money Pump Index(MPI) of a given dataset.
    MPI with cycle length of 2 is equivalent to the MPI with matrix form.
    In general, MPI is represented as the average of MPIs of all cycles in the dataset.
    
    Time complexity: NP 
    approx. O(n^m) when max_depth = m (m << n)
    '''
    G = _generate_graph(p, q)
    PQT = p @ q.T
    mci = []
    for i in nx.simple_cycles(G, max_depth):
        _sum_up = 0
        _sum_low = 0
        for k, v in (range(len(i)), 
            [len(i) - 1] + [i for i in range(len(i) - 1)]
        ):
            _sum_up += PQT[i[k], i[k]] - PQT[i[k], i[v]]
            _sum_low += PQT[i[k], i[k]]
        if _sum_up > 0:
            mci.append(_sum_up / _sum_low)

    e = np.mean(mci) if len(mci) > 0 else 0
    return e

def _matrix_mpi(p:np.ndarray, q:np.ndarray):
    '''
    The specialized version of MPI computation for the case of cycle length of 2.
    
    Time complexity: O(n^2)
    '''
    N = p.shape[0]
    PQT = p @ q.T
    ajM = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j and PQT[i, i] > PQT[i, j]:
                ajM[i, j] = 1
    cycleM = ajM * ajM.T
    mpi = np.zeros(int(np.sum(cycleM) / 2))
    t = 0
    for i in range(N):
        for j in range(N):
            if cycleM[i, j] == 1 and i > j:
                mpi[t] = 1 - (PQT[i, j] + PQT[j, i]) / (PQT[i, i] + PQT[j, j])
                t += 1

    mpi = mpi[mpi > 0]

    e = np.mean(mpi) if len(mpi) > 0 else 0
    return e
    
####################

mpi_cycle = _cycle_mpi
mpi_matrix = _matrix_mpi

__all__ = ['mpi_cycle', 'mpi_matrix']
