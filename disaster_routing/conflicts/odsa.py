import networkx as nx

from ..conflicts.check import check_dsa


def solve_odsa(graph: nx.Graph, perm: list[int], num_fses: list[int]) -> list[int]:
    result = [0 for _ in perm]
    for i, node in enumerate(perm):
        start = max(
            (
                result[prev] + num_fses[prev]
                for prev in perm[:i]
                if graph.has_edge(node, prev)
            ),
            default=0,
        )
        if len(result) > 0:
            start = max(start, result[-1])
        result[node] = start

    # check_dsa(graph, num_fses, result)
    return result
