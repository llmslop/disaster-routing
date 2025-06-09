import networkx as nx


def check_dsa(graph: nx.Graph, num_fses: list[int], result: list[int]):
    for i in range(len(graph)):
        for j in range(i + 1, len(graph)):
            ii = set(range(result[i], result[i] + num_fses[i]))
            ji = set(range(result[j], result[j] + num_fses[j]))
            if graph.has_edge(i, j):
                assert len(ii.intersection(ji)) == 0
    return 0
