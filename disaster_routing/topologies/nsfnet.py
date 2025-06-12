from .topology import DisasterZone, Topology
from statistics import mean

from .graphs import DiGraph


def nsfnet(enable_cross_edges: bool = False) -> Topology:
    graph = DiGraph()
    graph.add_nodes_from(range(1, 15, 1))
    assert len(graph.nodes) == 14

    edges = [
        (1, 2, 2100),
        (1, 3, 3000),
        (1, 9, 4800),
        (2, 3, 1200),
        (2, 4, 1500),
        (3, 6, 3600),
        (4, 5, 1200),
        (4, 11, 3900),
        (5, 7, 1200),
        (5, 6, 2400),
        (6, 10, 2100),
        (6, 14, 3600),
        (7, 8, 1500),
        (7, 10, 2700),
        (8, 9, 1500),
        (9, 10, 1500),
        (9, 12, 600),
        (9, 13, 600),
        (11, 12, 1200),
        (11, 13, 1500),
        (12, 14, 600),
        (13, 14, 300),
    ]

    edges = edges + [(b, a, w) for a, b, w in edges]
    graph.add_weighted_edges_from(edges)

    assert len(graph.edges) == 44, f"{len(graph.edges)} should be 44 (links)"

    avg_weight = graph.size("weight") / len(graph.edges)
    assert abs(avg_weight - 1936) < 1.0, f"{avg_weight} should be 1936 (km)"

    avg_degree = mean(graph.in_degree[i] for i in graph.nodes)
    assert abs(avg_degree - 3.14) < 0.01, f"{avg_degree} should be 3.14"

    dzs = [DisasterZone({node}) for node in graph]

    if enable_cross_edges:
        affected_edges = [
            (2, 1, 3),
            (5, 4, 11),
            (9, 11, 13),
            (9, 12, 14),
            (10, 6, 14),
            (12, 11, 13),
            (13, 12, 14),
        ]

        affected_edges = affected_edges + [(dz, b, a) for dz, a, b in affected_edges]
        for dzi, a, b in affected_edges:
            dzs[dzi - 1].edges.add((a, b))

    return Topology(graph, dzs)


if __name__ == "__main__":
    _ = nsfnet()
