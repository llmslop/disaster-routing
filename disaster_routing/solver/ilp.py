import logging
from collections import defaultdict
from typing import cast, override

from pulp import LpStatusNotSolved

from disaster_routing.eval.evaluator import Evaluator
from disaster_routing.ilp.cdp import ILPCDP
from disaster_routing.instances.instance import Instance
from disaster_routing.instances.modulation import ModulationFormat
from disaster_routing.routing.routing_algo import InfeasibleRouteError, Route
from disaster_routing.solver.solution import CDPSolution
from disaster_routing.solver.solver import CDPSolver
from disaster_routing.utils.ilist import ilist
from disaster_routing.utils.structlog import SL

log = logging.getLogger(__name__)


class ILPCDPSolver(CDPSolver):
    msg: bool
    time_limit: int | None

    def __init__(self, evaluator: Evaluator, msg: bool, time_limit: int | None = None):
        super().__init__(evaluator)
        self.msg = msg
        self.time_limit = time_limit

    @override
    def name(self) -> str:
        return "ilp"

    @staticmethod
    def float_eq(x: float, y: float, eps: float = 1e-6) -> bool:
        return abs(x - y) < eps

    @staticmethod
    def edge_list_to_node_list(edges: list[tuple[int, int]], start: int):
        adj: dict[int, list[int]] = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        path = [start]
        prev = None
        cur = start

        while True:
            nxt = next((n for n in adj[cur] if n != prev), None)
            if nxt is None:
                break
            path.append(nxt)
            prev, cur = cur, nxt

        return path

    @override
    def solve(
        self, inst: Instance, content_placement: dict[int, set[int]]
    ) -> CDPSolution:
        ilp = ILPCDP(inst, self.evaluator)
        solution = ilp.solve(self.msg, self.time_limit)
        if solution.status == LpStatusNotSolved:
            raise InfeasibleRouteError
        log.debug(
            SL(
                "ILP subobjectives",
                mofi=ilp.Delta.value(),
                total_fs=ilp.total_fs.value(),
            )
        )

        all_routes: list[ilist[Route]] = []
        start_indices: dict[int, int] = {}
        num_fses: dict[int, int] = {}
        route_idx = 0

        for r_idx, r in enumerate(inst.requests):
            routes: list[Route] = []
            for k in range(ilp.max_paths[r_idx]):
                arcs = [
                    a[:2]
                    for a_idx, a in enumerate(ilp.arcs)
                    if ILPCDPSolver.float_eq(
                        cast(float, ilp.p_k_ra[(k, r_idx, a_idx)].value()), 1
                    )
                ]
                Phi = cast(float, ilp.Phi_k_r[(k, r_idx)].value())
                w = cast(float, ilp.w_k_r[(k, r_idx)].value())
                g = cast(float, ilp.g_k_r[(k, r_idx)].value())
                node_list = ILPCDPSolver.edge_list_to_node_list(arcs, r.source)
                m = next(
                    (
                        m
                        for m_idx, m in enumerate(ModulationFormat.all())
                        if ILPCDPSolver.float_eq(
                            cast(float, ilp.b_k_mr[k, m_idx, r_idx].value()), 1
                        )
                    ),
                    None,
                )
                log.debug(
                    SL(
                        "ILP solution",
                        request=r.to_json(),
                        path=k,
                        w=w,
                        g=g,
                        m=m.name if m is not None else "none",
                        arcs=arcs,
                        node_list=node_list,
                        Phi=Phi,
                    )
                )
                if ILPCDPSolver.float_eq(w, 0):
                    continue

                start_indices[route_idx] = round(g)
                num_fses[route_idx] = round(Phi)
                assert m is not None
                routes.append(Route(inst.topology, tuple(node_list), m))
                route_idx += 1
            all_routes.append(tuple(routes))

        # NOTE: update content placement based on the ILP solution
        content_placement.clear()
        for content in ilp.contents:
            content_placement[content] = set()
            for d in ilp.avail_dcs:
                if ILPCDPSolver.float_eq(
                    cast(float, ilp.R_cr_d[(content, d)].value()), 1
                ):
                    content_placement[content].add(d)

        return CDPSolution(tuple(all_routes), start_indices, num_fses)
