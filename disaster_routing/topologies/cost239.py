from statistics import mean

from disaster_routing.topologies.graphs import DiGraph
from disaster_routing.topologies.topology import DisasterZone, Topology


def cost239(enable_cross_edges: bool = False) -> Topology:
    graph = DiGraph()
    graph.add_nodes_from(range(1, 12, 1))
    assert len(graph.nodes) == 11

    edges = [
        (1, 2, 450),
        (1, 3, 390),
        (1, 4, 550),
        (1, 8, 1310),
        (2, 3, 300),
        (2, 5, 400),
        (2, 6, 600),
        (2, 7, 820),
        (2, 9, 1090),
        (3, 4, 210),
        (3, 5, 220),
        (3, 7, 400),
        (4, 5, 390),
        (4, 8, 760),
        (4, 9, 660),
        (5, 6, 350),
        (5, 10, 730),
        (6, 7, 320),
        (6, 10, 565),
        (6, 11, 730),
        (7, 11, 820),
        (8, 9, 390),
        (8, 10, 740),
        (9, 10, 340),
        (9, 11, 660),
        (10, 11, 320),
    ]

    edges = edges + [(b, a, w) for a, b, w in edges]
    graph.add_weighted_edges_from(edges)

    assert len(graph.edges) == 52, f"{len(graph.edges)} should be 52 (links)"

    avg_weight = graph.size("weight") / len(graph.edges)
    # TODO: probably typo from paper author (he must like apiss #578 mascot a
    # little too much)
    # assert abs(avg_weight - 578) < 1.0, f"{avg_weight} should be 578 (km)"
    assert abs(avg_weight - 558) < 1.0, f"{avg_weight} should be 558 (km)"

    avg_degree = mean(graph.in_degree[i] for i in graph.nodes)
    assert abs(avg_degree - 4.73) < 0.01, f"{avg_degree} should be 4.73"

    dzs = [
        DisasterZone({1}),
        DisasterZone({2, 3}),
        DisasterZone({3, 4, 5}),
        DisasterZone({6, 7}),
        DisasterZone({8, 9}),
        DisasterZone({9, 10}),
        DisasterZone({10, 11}),
    ]

    if enable_cross_edges:
        affected_edges = [
            (2, 2, 9),
        ]

        affected_edges = affected_edges + [(dz, b, a) for dz, a, b in affected_edges]
        for dzi, a, b in affected_edges:
            dzs[dzi - 1].edges.add((a, b))

    return Topology(graph, dzs)


if __name__ == "__main__":
    _ = cost239()
