"""Dependency decoding: single-root maximum spanning arborescence.

Chu--Liu/Edmonds (via networkx) over the biaffine arc scores, constrained so exactly one
node attaches to the artificial ROOT (node 0). This guarantees a valid single-root tree
(no cycles, one root) for conll18_ud_eval. Moved verbatim from predict.py.
"""
import networkx as nx


def mst_decode(arc, n):
    """Maximum spanning arborescence rooted at node 0 (Chu-Liu-Edmonds via nx).
    arc[d][h] = score of head h -> dependent d. Returns heads[0..n], heads[0]=0,
    guaranteeing a valid single-root tree (no cycles) for conll18_ud_eval."""
    if n <= 1:
        return [0] * (n + 1)
    # single root: only the best root-candidate may attach to node 0
    root = max(range(1, n + 1), key=lambda i: float(arc[i][0]))
    G = nx.DiGraph()
    G.add_nodes_from(range(n + 1))
    for d in range(1, n + 1):
        for h in range(0, n + 1):
            if h == d:
                continue
            if h == 0 and d != root:
                continue                       # forbid extra roots
            G.add_edge(h, d, weight=float(arc[d][h]))
    arb = nx.maximum_spanning_arborescence(G, attr="weight", preserve_attrs=False)
    heads = [0] * (n + 1)
    for h, d in arb.edges():
        heads[d] = h
    return heads
