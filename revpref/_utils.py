import networkx as nx
import pulp as pl
import numpy as np

def simple_cycles(G: nx.Graph, max_depth = 0):
    r'''
    Johnson's algorithm (1975)
    Find all cycles in graph with max depth
    We modified it with max depth
    '''
    if max_depth <= 0:
        max_depth = len(G)
    subG = G.copy()
    sccs: list = list(nx.strongly_connected_components(subG))
    while sccs:
        scc: list = sccs.pop()
        startnode = scc.pop()
        
        path = [startnode]
        blocked = set()
        blocked.add(startnode)
        stack = [(startnode, list(subG[startnode]))]

        while stack:
            thisnode, nbrs = stack[-1]
            if nbrs and len(path) <= max_depth:
                nextnode = nbrs.pop()
                if nextnode == startnode:
                    yield path[:]
                elif nextnode not in blocked:
                    path.append(nextnode)
                    stack.append((nextnode, list(subG[nextnode])))
                    blocked.add(nextnode)
                    continue
            if not nbrs or len(path) >= max_depth:
                blocked.remove(thisnode)
                stack.pop()
                path.pop()

        subG.remove_node(startnode)
        H = subG.subgraph(scc)
        sccs.extend(list(nx.strongly_connected_components(H)))
        
nx.simple_cycles = simple_cycles

def generate_graph(p:np.ndarray, q:np.ndarray, e = 1):
    N = p.shape[0]
    G = nx.DiGraph()
    PQT = p @ q.T
    for i in range(N):
        G.add_node(i)
    for i in range(N):
        for j in range(N):
            if i != j:
                if e * (PQT[i, i]) > (PQT[i, j]):
                    G.add_edge(i, j, weight = e * (PQT[i, i]) - (PQT[i, j]))
    return G
    
def has_cycle(G):
    try:
        nx.find_cycle(G)
        return True
    except:
        return False