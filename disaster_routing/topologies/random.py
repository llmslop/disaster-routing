from math import sqrt
from random import random
from typing import cast
import networkx as nx

from disaster_routing.topologies.topology import DisasterZone, Topology, make_digraph
from disaster_routing.topologies.graphs import Graph


def dist(p1: tuple[float, float], p2: tuple[float, float]):
    x, y = p1[0] - p2[0], p1[1] - p2[1]
    return sqrt(x**2 + y**2)


def random_topology(n: int = 20, beta: float = 0.4, alpha: float = 0.5) -> Topology:
    graph = cast(Graph, nx.waxman_graph(n, beta, alpha))
    assert len(graph.nodes) == n

    for u, v in graph.edges:
        graph.edges[u, v]["weight"] = (
            dist(graph.nodes[u]["pos"], graph.nodes[v]["pos"]) * 1000.0
        )

    # pos = [graph.nodes[node]["pos"] for node in graph]
    rem_nodes = set(range(n))
    dzs: list[DisasterZone] = []
    while len(rem_nodes) > 0:
        u, v, r = random(), random(), 0.2
        dz_nodes = {
            node for node in rem_nodes if dist(graph.nodes[node]["pos"], (u, v)) < r
        }
        rem_nodes.difference_update(dz_nodes)

        dzs.append(DisasterZone(dz_nodes))

    return Topology(make_digraph(graph), dzs)


if __name__ == "__main__":
    _ = random_topology()
