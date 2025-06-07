from typing import cast
from scipy import optimize as opt

import numpy as np
import networkx as nx
from scipy.sparse import dok_matrix


from ..topologies.topology import Topology
from ..instances.request import Request
from ..instances.instance import Instance


def solve_dc_placement_single_request(
    dcs: list[int], graph: nx.DiGraph, requests: list[Request]
) -> list[int]:
    total_lengths = {node: 0 for node in graph}
    for req in requests:
        lengths: dict[int, int] = nx.single_source_dijkstra_path_length(
            graph,
            cast(str, cast(object, req.source)),
            weight=None,
        )
        for node, length in lengths.items():
            total_lengths[node] += length

    dc_lengths = [total_lengths[dc] for dc in dcs]

    # ILP variables:
    # gamma^k_rd => gamma[d][r][k]
    # R^(c_r)_d => R[d]
    # We construct a `d K + d`-vector, first components are R[d]'s, last
    # components are gammas, where K = sum(req.max_path_count for req in
    # requests)

    # The number of constraints:
    K = sum(req.max_path_count for req in requests)
    D = len(dcs)
    R = len(requests)

    num_constraints = [K, 1, D * R]
    num_components = D * (K + 1)

    A = dok_matrix((sum(num_constraints), num_components))
    lb = np.zeros(sum(num_constraints))
    ub = np.ones(sum(num_constraints))
    constraint = 0
    for i in range(num_constraints[0]):
        A[constraint, i : D * K : K] = 1
        lb[constraint] = ub[constraint] = 1
        constraint += 1
    for _ in range(num_constraints[1]):
        A[constraint, D * K : -1] = 1
        lb[constraint] = 2
        ub[constraint] = min(req.max_path_count for req in requests)
        constraint += 1
    # for i in range(num_constraints[2]):
    start = 0
    for r in range(R):
        end = start + requests[r].max_path_count
        for d in range(D):
            A[constraint, start + d * K : end + d * K] = 1
            A[constraint, D * K + d] = -1
            lb[constraint] = -np.inf
            ub[constraint] = 0
            constraint += 1
        start = end

    bounds = opt.Bounds(0, 1)
    constraints = opt.LinearConstraint(A, lb, ub)
    c = np.zeros(num_components)
    for d in range(D):
        c[D * K + d] = dc_lengths[d]

    result = opt.milp(
        c, integrality=np.ones(num_components), bounds=bounds, constraints=constraints
    )

    assert result.success, "Unable to find feasible content placement"
    return [d for d in range(D) if result.x[d + D * K] == 1]


def solve_dc_placement(inst: Instance, dcs: list[int]) -> list[list[int]]:
    contents = set(req.content_id for req in inst.requests)
    result: list[list[int]] = [[] for _ in dcs]
    for content in contents:
        requests = [req for req in inst.requests if req.content_id == content]
        dcs_with_content = solve_dc_placement_single_request(
            dcs, inst.topology.graph, requests
        )
        for dc in dcs_with_content:
            result[dc].append(content)
    return result
