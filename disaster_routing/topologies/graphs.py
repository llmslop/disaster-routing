from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    Graph = nx.Graph[int]
    DiGraph = nx.DiGraph[int]
    StrGraph = nx.Graph[str]
    StrDiGraph = nx.DiGraph[str]
else:
    Graph = nx.Graph
    DiGraph = nx.DiGraph
    StrGraph = Graph
    StrDiGraph = DiGraph
