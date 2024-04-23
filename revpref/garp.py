__version__ = '1.0.0'
__author__ = 'Yi Xiang'

import networkx as nx
import numpy as np
from typing import List, Tuple, Literal
from matplotlib import pyplot as plt

from ._utils import generate_graph

from .ccei import _warshall_ccei, _dichotomy_ccei, _mtz_ccei
from .mpi import _cycle_mpi, _matrix_mpi
from .mci import _mtz_mci, _optimize_mci, _milp_mci
from .hmi import _mtz_hmi, _gross_hmi
from .avi import _mtz_avi

def _generate_graph(p, q, e = 1):
    return generate_graph(p, q, e)

class RevealedPreference:
    '''
    RevealedPreference
    =================
    >>> import revpref
    >>> bf = revpref.RevealedPreference(p, q)
    >>> bf.check_garp()
    >>> bf.ccei()
    ...
    =================
    p : np.ndarray
        The matrix of prices
    q : np.ndarray
        The matrix of quantities
    e : float
        The Afrait efficiency (default 1)
    '''
    p = np.array([])
    q = np.array([])
    e = 1
    G : nx.DiGraph = None
    
    def __init__(self, p, q, efficiency = 1):
        self.p = np.array(p, dtype=np.float64)
        self.q = np.array(q, dtype=np.float64)
        self.e = efficiency
        if self.p.shape != self.q.shape:
            raise ValueError("shape of p and q should be the same")
        if len(self.p.shape) != 2:
            raise ValueError("p and q should be 2-dim arrays")
        self.G = _generate_graph(self.p, self.q, efficiency)
    
    def check_garp(self):
        st = True
        for i in nx.simple_cycles(self.G):
            if len(i) > 1:
                st = False
                break
        return st
    
    def ccei(self, 
        method : Literal['warshall', 'dichotomy', 'milp'] = 'warshall',
        tol : float = 1e-7,
        lp_solver : Literal[
            'GUROBI_CMD', 'PULP_CBC_CMD', 'COIN_CMD',
            'CPLEX_CMD', 'GLPK_CMD', 'XPRESS', 'CPLEX_PY'
        ] = 'PULP_CBC_CMD',
        jit : bool = False
    ):
        '''
        CCEI (Critical Cost Efficiency Index)
        -- Afriat (1972)
        
        Our package provides three methods to calculate CCEI:
        1. Warshall Algorithm (default, suggested)
        2. Dichotomy
        3. MTZ
        '''
        match method:
            case 'warshall':
                return _warshall_ccei(self.p, self.q)
            case 'dichotomy':
                return _dichotomy_ccei(self.p, self.q, tol)
            case 'mtz':
                return _mtz_ccei(self.p, self.q, lp_solver)
        raise ValueError("method should be 'warshall' or 'dichotomy' or 'mtz'")
    
    def avi(self, 
        method : Literal['mtz'] = 'mtz',
        lp_solver : Literal[
            'GUROBI_CMD', 'PULP_CBC_CMD', 'COIN_CMD',
            'CPLEX_CMD', 'GLPK_CMD', 'XPRESS', 'CPLEX_PY'
        ] = 'PULP_CBC_CMD'
    ):
        match method:
            case 'mtz':
                return _mtz_avi(self.p, self.q, lp_solver)
        raise ValueError('x') 
    
    def violations(self):
        return nx.simple_cycles(self.G)
    
    def mpi(self,
        method : Literal['cycle', 'matrix'] = 'cycle', 
        max_depth : int = 2,
        jit : bool = False
    ):
        match method:
            case 'cycle':
                return _cycle_mpi(self.p, self.q, max_depth)
            case 'matrix':
                if max_depth != 2:
                    raise ValueError("max_depth should be 2 for matrix method")
                return _matrix_mpi(self.p, self.q)
        raise ValueError("method should be 'cycle' or 'matrix'")
    
    def mci(self, 
        method : Literal['optimize', 'milp', 'mtz'] = 'optimize',
        lp_solver : Literal[
            'GUROBI_CMD', 'PULP_CBC_CMD', 'COIN_CMD',
            'CPLEX_CMD', 'GLPK_CMD', 'XPRESS', 'CPLEX_PY'
        ] = 'PULP_CBC_CMD'
    ):
        match method:
            case 'milp':
                return _milp_mci(self.p, self.q)
            case 'mtz':
                return _mtz_mci(self.p, self.q)
            case 'optimize':
                return _optimize_mci(self.p, self.q)
        raise ValueError("method should be 'optimize', 'milp' or 'mtz'")
    
    def hmi(self,
        method : Literal['mtz', 'gross'] = 'mtz',
        lp_solver : Literal[
            'GUROBI_CMD', 'PULP_CBC_CMD', 'COIN_CMD',
            'CPLEX_CMD', 'GLPK_CMD', 'XPRESS', 'CPLEX_PY'
        ] = 'PULP_CBC_CMD'
    ):
        match method:
            case 'mtz':
                return _mtz_hmi(self.p, self.q, lp_solver)
            case 'gross':
                return _gross_hmi(self.p, self.q)
        raise ValueError("method should be 'mtz' or 'gross'")
    
    def draw(self):
        G = self.G
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        p = plt.figure(figsize=(8, 8))
        nx.draw_networkx_nodes(
            G, pos=nx.circular_layout(G), 
            node_size=600, node_color='white', 
            edgecolors='grey', linewidths=0.9
        )
        nx.draw_networkx_edges(
            G, pos=nx.circular_layout(G), 
            edge_color='blue', width=0.5, min_target_margin=12, 
            min_source_margin=0.1, connectionstyle='arc3,rad=0.005'
        )
        nx.draw_networkx_labels(
            G, pos=nx.circular_layout(G), 
        font_size=12)
        plt.axis('off')
        return p
    
    def show(self):
        self.draw()
        plt.show()
        return self
        
    def graph(self):
        return self.G