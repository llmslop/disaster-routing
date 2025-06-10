import networkx as nx

from .odsa import solve_odsa
from .mofi import calc_mofi
from .check import check_dsa


def fpga_with_first(graph: nx.Graph, first: int, num_fses: list[int]) -> list[int]:
    ordering = [first]
    remaining = set(range(len(graph))).difference(ordering)

    start_indices = [0 for _ in range(len(graph))]

    while len(remaining) > 0:
        node = min(
            remaining,
            key=lambda node: max(
                (
                    start_indices[prev] + num_fses[prev]
                    for prev in ordering
                    if graph.has_edge(node, prev)
                ),
                default=0,
            ),
        )
        start_indices[node] = max(
            (
                start_indices[prev] + num_fses[prev]
                for prev in ordering
                if graph.has_edge(node, prev)
            ),
            default=0,
        )

        remaining.remove(node)
        ordering.append(node)

    check_dsa(graph, num_fses, start_indices)
    return ordering


def fpga(graph: nx.Graph, num_fses: list[int]) -> list[int]:
    return min(
        (fpga_with_first(graph, start, num_fses) for start in graph),
        key=lambda sol: calc_mofi(num_fses, sol),
    )


def fpga_direct(graph: nx.Graph, num_fses: list[int]) -> list[int]:
    perm = fpga(graph, num_fses)
    return solve_odsa(graph, perm, num_fses)
