from statistics import mean

from disaster_routing.topologies.graphs import DiGraph
from disaster_routing.topologies.topology import DisasterZone, Topology


def us_backbone() -> Topology:
    graph = DiGraph()
    graph.add_nodes_from(range(1, 29, 1))
    assert len(graph.nodes) == 28

    edges = [
        (1, 2, 768),
        (1, 4, 608),
        (2, 3, 528),
        (2, 5, 528),
        (3, 5, 672),
        (3, 8, 640),
        (4, 6, 640),
        (4, 7, 640),
        (5, 7, 368),
        (6, 9, 512),
        (7, 8, 992),
        (7, 11, 848),
        (8, 13, 928),
        (9, 10, 336),
        (9, 11, 256),
        (10, 14, 240),
        (11, 12, 320),
        (11, 14, 560),
        (12, 13, 736),
        (12, 15, 440),
        (13, 16, 320),
        (13, 18, 960),
        (14, 15, 272),
        (14, 17, 240),
        (14, 23, 366),
        (15, 17, 228),
        (16, 19, 512),
        (17, 18, 256),
        (17, 20, 576),
        (17, 23, 256),
        (18, 19, 464),
        (18, 20, 576),
        (19, 21, 432),
        (19, 22, 528),
        (20, 21, 400),
        (20, 25, 320),
        (21, 22, 272),
        (23, 24, 352),
        (23, 26, 320),
        (23, 28, 288),
        (24, 25, 336),
        (24, 27, 416),
        (24, 28, 176),
        (26, 27, 336),
        (27, 28, 240),
    ]

    edges = edges + [(b, a, w) for a, b, w in edges]
    graph.add_weighted_edges_from(edges)

    assert len(graph.edges) == 90, f"{len(graph.edges)} should be 52 (links)"

    avg_weight = graph.size("weight") / len(graph.edges)
    assert abs(avg_weight - 466) < 1.0, f"{avg_weight} should be 558 (km)"

    avg_degree = mean(graph.in_degree[i] for i in graph.nodes)
    assert abs(avg_degree - 3.2) < 0.1, f"{avg_degree} should be 4.73"

    dzs = [
        DisasterZone({1}),
        DisasterZone({2}),
        DisasterZone({3}),
        DisasterZone({4}),
        DisasterZone({5, 7}),
        DisasterZone({6}),
        DisasterZone({8}),
        DisasterZone({9, 10}),
        DisasterZone({11, 12}),
        DisasterZone({13, 16}),
        DisasterZone({14, 15, 17, 23}),
        DisasterZone({18, 19}),
        DisasterZone({20, 25}),
        DisasterZone({21, 22}),
        DisasterZone({24, 26, 27, 28}),
    ]

    assert all(any(node in dz.nodes for dz in dzs) for node in graph.nodes)

    return Topology(graph, dzs)


if __name__ == "__main__":
    _ = us_backbone()
