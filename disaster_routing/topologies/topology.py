from copy import deepcopy
from typing import cast
from networkx.readwrite.json_graph import node_link_data, node_link_graph

from ..utils.ilist import ilist
from .graphs import Graph, DiGraph


def make_digraph(graph: Graph) -> DiGraph:
    """Converts an undirected NetworkX graph to a directed graph."""
    digraph = DiGraph()

    # Copy nodes and their attributes
    digraph.add_nodes_from(graph.nodes(data=True))

    # For each edge (u, v), add both (u, v) and (v, u) to the DiGraph
    for u, v, data in graph.edges(data=True):
        digraph.add_edge(u, v, **data)
        digraph.add_edge(v, u, **data)

    return digraph


class DisasterZone:
    nodes: set[int]
    edges: set[tuple[int, int]]

    def __init__(
        self, nodes: set[int] | None = None, edges: set[tuple[int, int]] | None = None
    ):
        self.nodes = set(nodes) if nodes is not None else set()
        self.edges = set(edges) if edges is not None else set()

    def affects_path(self, path: ilist[int], exclude_source: bool = False):
        path_nodes = path if not exclude_source else path[1:]
        if any(node in self.nodes for node in path_nodes):
            return True

        edges = ((path[i], path[i + 1]) for i in range(len(path) - 1))
        return any(edge in self.edges for edge in edges)

    def remove_from_graph(self, g: DiGraph):
        g.remove_nodes_from(self.nodes)
        g.remove_edges_from(self.edges)

    @staticmethod
    def from_json(data: dict[str, object]) -> "DisasterZone":
        return DisasterZone(
            set(cast(list[int], data["nodes"])),
            set((edge[0], edge[1]) for edge in cast(list[list[int]], data["edges"])),
        )

    def to_json(self) -> object:
        return {
            "nodes": list(self.nodes),
            "edges": list(self.edges),
        }


class Topology:
    graph: DiGraph
    dzs: list[DisasterZone]

    def __init__(self, graph: DiGraph, dzs: list[DisasterZone]):
        self.graph = graph
        self.dzs = deepcopy(dzs)

    def copy(self) -> "Topology":
        return Topology(self.graph.copy(), deepcopy(self.dzs))

    @staticmethod
    def from_json(data: dict[str, object]) -> "Topology":
        return Topology(
            node_link_graph(data["graph"], edges="links"),
            [
                DisasterZone.from_json(dz)
                for dz in cast(list[dict[str, object]], data["dzs"])
            ],
        )

    def to_json(self) -> object:
        return cast(
            object,
            {
                "graph": node_link_data(self.graph, edges="links"),
                "dzs": [dz.to_json() for dz in self.dzs],
            },
        )
