from copy import deepcopy
import networkx as nx


class DisasterZone:
    nodes: set[int]
    edges: set[tuple[int, int]]

    def __init__(self, nodes: set[int] = set(), edges: set[tuple[int, int]] = set()):
        self.nodes = set(nodes)
        self.edges = set(edges)

    def affects_path(self, path: list[int], exclude_source=False):
        path_nodes = path if not exclude_source else path[1:]
        if any(node in self.nodes for node in path_nodes):
            return True

        edges = ((path[i], path[i + 1]) for i in range(len(path) - 1))
        return any(edge in self.edges for edge in edges)

    def remove_from_graph(self, g: nx.DiGraph):
        g.remove_nodes_from(self.nodes)
        g.remove_edges_from(self.edges)


class Topology:
    graph: nx.DiGraph
    dzs: list[DisasterZone]

    def __init__(self, graph: nx.DiGraph, dzs: list[DisasterZone]):
        self.graph = graph
        self.dzs = deepcopy(dzs)
