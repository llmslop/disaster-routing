import logging
from math import ceil
from typing import cast, final

from pulp import (
    PULP_CBC_CMD,
    LpAffineExpression,
    LpBinary,
    LpConstraint,
    LpInteger,
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    lpSum,
)

from disaster_routing.eval.evaluator import Evaluator
from disaster_routing.instances.instance import Instance
from disaster_routing.instances.modulation import ModulationFormat
from disaster_routing.routing.routing_algo import Route
from disaster_routing.utils.ilist import ilist
from disaster_routing.utils.structlog import SL

log = logging.getLogger(__name__)


class ILPCDPSolution:
    status: int
    var_values: dict[str, float]
    objective: float | None

    def __init__(
        self, status: int, var_values: dict[str, float], objective: float | None
    ) -> None:
        self.status = status
        self.var_values = var_values
        self.objective = objective

    def status_str(self):
        return LpStatus[self.status]


@final
class ILPCDP:
    def __init__(self, inst: Instance, evaluator: Evaluator) -> None:
        S = 1000
        max_paths = [
            int(inst.topology.graph.in_degree[req.source]) for req in inst.requests
        ]

        self.max_paths = max_paths

        log.debug(SL("Construct ILP for CDP"))

        arcs: list[tuple[int, int, int]] = [
            (u, v, inst.topology.graph.edges[u, v]["weight"])
            for u, v in inst.topology.graph.edges
        ]

        self.inst = inst
        avail_dcs = inst.possible_dc_positions
        self.avail_dcs = avail_dcs
        self.arcs = arcs

        self.p_k_ra = {
            (k, r_idx, a_idx): LpVariable(f"p^{k}_{r_idx}_{a_idx}", cat=LpBinary)
            for r_idx, _ in enumerate(inst.requests)
            for a_idx, _ in enumerate(arcs)
            for k in range(max_paths[r_idx])
        }

        self.Lambda_k_rd = {
            (k, r_idx, d): LpVariable(f"Λ^{k}_{r_idx}_{d}", cat=LpBinary)
            for r_idx, _ in enumerate(inst.requests)
            for d in avail_dcs
            for k in range(max_paths[r_idx])
        }

        self.alpha_k_rz = {
            (k, r_idx, z_idx): LpVariable(f"α^{k}_{r_idx}_{z_idx}", cat=LpBinary)
            for r_idx, _ in enumerate(inst.requests)
            for z_idx, _ in enumerate(inst.topology.dzs)
            for k in range(max_paths[r_idx])
        }

        contents = list({req.content_id for req in inst.requests})
        self.contents = contents
        self.R_cr_d = {
            (cr, d): LpVariable(f"R^{cr}_{d}", cat=LpBinary)
            for cr in contents
            for d in avail_dcs
        }

        self.Phi_k_rm = {
            (k, r_idx, m_idx): LpVariable(
                f"Φm^{k}_{r_idx}_{m_idx}", cat=LpInteger, lowBound=0, upBound=S
            )
            for r_idx, r in enumerate(inst.requests)
            for m_idx, m in enumerate(ModulationFormat.all())
            for k in range(max_paths[r_idx])
        }

        self.w_k_r = {
            (k, r_idx): LpVariable(f"w^{k}_{r_idx}", cat=LpBinary)
            for r_idx, _ in enumerate(inst.requests)
            for k in range(max_paths[r_idx])
        }

        self.xi_i_r = {
            (i, r_idx): LpVariable(f"ξ^{i}_{r_idx}", cat=LpBinary)
            for r_idx, _ in enumerate(inst.requests)
            for i in range(1, max_paths[r_idx])
        }

        self.Phi_k_ra = {
            (k, r_idx, a_idx): LpVariable(
                f"Φa^{k}_{r_idx}_{a_idx}", cat=LpInteger, lowBound=0, upBound=S
            )
            for r_idx, _ in enumerate(inst.requests)
            for a_idx, _ in enumerate(arcs)
            for k in range(max_paths[r_idx])
        }

        self.Phi_k_r = {
            (k, r_idx): LpVariable(
                f"Φ^{k}_{r_idx}", cat=LpInteger, lowBound=0, upBound=S
            )
            for r_idx, _ in enumerate(inst.requests)
            for k in range(max_paths[r_idx])
        }

        self.g_k_r = {
            (k, r_idx): LpVariable(
                f"g^{k}_{r_idx}", cat=LpInteger, lowBound=0, upBound=S - 1
            )
            for r_idx, _ in enumerate(inst.requests)
            for k in range(max_paths[r_idx])
        }

        self.beta_kkp_r = {
            (k, k_prime, r_idx): LpVariable(f"β^{k}^{k_prime}_{r_idx}", cat=LpBinary)
            for r_idx, _ in enumerate(inst.requests)
            for k in range(max_paths[r_idx])
            for k_prime in range(max_paths[r_idx])
        }

        self.beta_kkp_rrp: dict[tuple[int, int, int, int], LpVariable] = {
            (k, k_prime, r_idx, r_idx_prime): LpVariable(
                f"β^{k}^{k_prime}_{r_idx}_{r_idx_prime}", cat=LpBinary
            )
            for r_idx, _ in enumerate(inst.requests)
            for r_idx_prime, _ in enumerate(inst.requests)
            for k in range(max_paths[r_idx])
            for k_prime in range(max_paths[r_idx_prime])
        }

        self.gamma_kkp_r = {
            (k, k_prime, r_idx): LpVariable(f"γ^{k}^{k_prime}_{r_idx}", cat=LpBinary)
            for r_idx, _ in enumerate(inst.requests)
            for k in range(max_paths[r_idx])
            for k_prime in range(max_paths[r_idx])
        }

        self.gamma_kkp_rrp = {
            (k, k_prime, r_idx, r_idx_prime): LpVariable(
                f"γ^{k}^{k_prime}_{r_idx}_{r_idx_prime}", cat=LpBinary
            )
            for r_idx, _ in enumerate(inst.requests)
            for r_idx_prime, _ in enumerate(inst.requests)
            if r_idx != r_idx_prime
            for k in range(max_paths[r_idx])
            for k_prime in range(max_paths[r_idx_prime])
        }

        self.b_k_mr = {
            (k, m_idx, r_idx): LpVariable(f"b^{k}_{m_idx}_{r_idx}", cat=LpBinary)
            for r_idx, r in enumerate(inst.requests)
            for m_idx, m in enumerate(ModulationFormat.all())
            for k in range(max_paths[r_idx])
        }

        self.theta_d = {
            d: LpVariable(f"θ_{d}", cat=LpBinary) for d in inst.possible_dc_positions
        }

        self.Delta = LpVariable("Δ", cat=LpInteger, lowBound=0, upBound=S)

        self.problem = LpProblem("CDP", LpMinimize)

        weights = evaluator.get_weights()
        if weights is None:
            raise ValueError(
                "Evaluator must provide weights for the objective function"
            )
        theta_1, theta_2 = weights
        total_fs = lpSum(
            self.Phi_k_ra[k, r_idx, a_idx]
            for r_idx, _ in enumerate(inst.requests)
            for a_idx, _ in enumerate(arcs)
            for k in range(max_paths[r_idx])
        )

        self.problem += (theta_1 * total_fs + theta_2 * self.Delta, "Objective")
        self.total_fs = total_fs

        # constraints
        self.problem += (
            lpSum(self.theta_d[d] for d in inst.possible_dc_positions) <= inst.dc_count,
            "DC count constraint (0.1)",
        )

        for d in inst.possible_dc_positions:
            for cr in contents:
                self.problem += (
                    self.R_cr_d[(cr, d)] <= self.theta_d[d],
                    f"DC count constraint (0.2:{cr}:{d}",
                )

        for r_idx, r in enumerate(inst.requests):
            for k in range(max_paths[r_idx]):
                self.problem += (
                    lpSum(self.Lambda_k_rd[(k, r_idx, d)] for d in avail_dcs)
                    == self.w_k_r[(k, r_idx)],
                    f"DC assignment and content placement constraints (2:{r_idx}:{k})",
                )

        for r_idx, r in enumerate(inst.requests):
            sm = lpSum(self.R_cr_d[(r.content_id, d)] for d in avail_dcs)
            self.problem += (
                sm >= 2,
                f"DC assignment and content placement constraints (3.1:{r_idx})",
            )
            self.problem += (
                sm <= max_paths[r_idx],
                f"DC assignment and content placement constraints (3.2:{r_idx})",
            )

        for r_idx, r in enumerate(inst.requests):
            for d in avail_dcs:
                self.problem += (
                    lpSum(
                        self.Lambda_k_rd[(k, r_idx, d)] for k in range(max_paths[r_idx])
                    )
                    <= self.R_cr_d[(r.content_id, d)],
                    f"DC assignment and content placement constraints (4:{r_idx}:{d})",
                )

        for r_idx, r in enumerate(inst.requests):
            for k in range(max_paths[r_idx]):
                for v in inst.topology.graph.nodes:
                    Psi_v_plus = {i for i, a in enumerate(arcs) if a[0] == v}
                    Psi_v_minus = {i for i, a in enumerate(arcs) if a[1] == v}

                    tar: LpAffineExpression | LpVariable | int = 0
                    if v == r.source:
                        tar = self.w_k_r[(k, r_idx)]
                    elif v in avail_dcs:
                        tar = -self.Lambda_k_rd[(k, r_idx, v)]

                    self.problem += (
                        lpSum(self.p_k_ra[(k, r_idx, a_idx)] for a_idx in Psi_v_plus)
                        - lpSum(self.p_k_ra[(k, r_idx, a_idx)] for a_idx in Psi_v_minus)
                        == tar,
                        f"Flow-conservation constraints (5:{r_idx}:{k}:{v})",
                    )

        dz_arcs = {
            z_idx: [
                i
                for i, a in enumerate(arcs)
                if any(edge == a[:1] for edge in z.edges)
                or any(n == a[0] or n == a[1] for n in z.nodes)
            ]
            for z_idx, z in enumerate(inst.topology.dzs)
        }
        self.dz_arcs = dz_arcs

        log.debug(SL("Disaster zone arcs", dz_arcs=dz_arcs))

        for r_idx, r in enumerate(inst.requests):
            for z_idx, _ in enumerate(inst.topology.dzs):
                for k in range(max_paths[r_idx]):
                    self.problem += (
                        lpSum(
                            self.p_k_ra[(k, r_idx, a_idx)] for a_idx in dz_arcs[z_idx]
                        )
                        >= self.alpha_k_rz[(k, r_idx, z_idx)],
                        "Disaster-zone-disjoint path constraints "
                        + f"(6:{r_idx}:{z_idx}:{k})",
                    )

        for r_idx, r in enumerate(inst.requests):
            for z_idx, _ in enumerate(inst.topology.dzs):
                for k in range(max_paths[r_idx]):
                    for a_idx in dz_arcs[z_idx]:
                        self.problem += (
                            self.alpha_k_rz[(k, r_idx, z_idx)]
                            >= self.p_k_ra[(k, r_idx, a_idx)],
                            "Disaster-zone-disjoint path constraints "
                            + f"(7:{r_idx}:{z_idx}:{k}:{a_idx})",
                        )

        for r_idx, r in enumerate(inst.requests):
            for z_idx, z in enumerate(inst.topology.dzs):
                if r.source in z.nodes:
                    continue
                self.problem += (
                    lpSum(
                        self.alpha_k_rz[(k, r_idx, z_idx)]
                        for k in range(max_paths[r_idx])
                    )
                    <= 1,
                    f"Disaster-zone-disjoint path constraints (8:{r_idx}:{z_idx})",
                )

        for r_idx, r in enumerate(inst.requests):
            for k in range(max_paths[r_idx]):
                for m_idx, m in enumerate(ModulationFormat.all()):
                    self.problem += (
                        lpSum(
                            a[2] * self.p_k_ra[(k, r_idx, a_idx)]
                            for a_idx, a in enumerate(arcs)
                        )
                        <= m.reach
                        + ModulationFormat.bpsk().reach
                        * (1 - self.b_k_mr[(k, m_idx, r_idx)]),
                        f"Modulation adaption constraints (9:{r_idx}:{k}:{m_idx})",
                    )

        for r_idx, r in enumerate(inst.requests):
            for k in range(max_paths[r_idx]):
                self.problem += (
                    lpSum(
                        self.b_k_mr[(k, m_idx, r_idx)]
                        for m_idx, _ in enumerate(ModulationFormat.all())
                    )
                    == self.w_k_r[(k, r_idx)],
                    f"Modulation adaption constraints (10:{r_idx}:{k})",
                )

        for r_idx, r in enumerate(inst.requests):
            for k in range(max_paths[r_idx]):
                self.problem += (
                    lpSum(
                        self.Phi_k_rm[(k, r_idx, m_idx)]
                        for m_idx, _ in enumerate(ModulationFormat.all())
                    )
                    == self.Phi_k_r[(k, r_idx)],
                    f"Modulation adaption constraints (11:{r_idx}:{k})",
                )

        for r_idx, r in enumerate(inst.requests):
            for k in range(max_paths[r_idx]):
                for m_idx, m in enumerate(ModulationFormat.all()):
                    self.problem += (
                        self.Phi_k_rm[(k, r_idx, m_idx)]
                        <= S * self.b_k_mr[(k, m_idx, r_idx)],
                        f"Modulation adaption constraints (12:{r_idx}:{k}:{m_idx})",
                    )

        for r_idx, r in enumerate(inst.requests):
            for k in range(max_paths[r_idx]):
                for a_idx, a in enumerate(arcs):
                    self.problem += (
                        self.p_k_ra[(k, r_idx, a_idx)] <= self.w_k_r[(k, r_idx)],
                        "Adaptive multi-path routing constraints "
                        + f"(13:{r_idx}:{k}:{a_idx})",
                    )
        for r_idx, r in enumerate(inst.requests):
            self.problem += (
                lpSum(self.w_k_r[(k, r_idx)] for k in range(max_paths[r_idx] - 1))
                == lpSum(
                    k * self.xi_i_r[(k, r_idx)] for k in range(1, max_paths[r_idx])
                ),
                f"Adaptive multi-path routing constraints (14:{r_idx})",
            )
        for r_idx, r in enumerate(inst.requests):
            self.problem += (
                lpSum(self.xi_i_r[(k, r_idx)] for k in range(1, max_paths[r_idx])) <= 1,
                f"Adaptive multi-path routing constraints (15:{r_idx})",
            )
        for r_idx, r in enumerate(inst.requests):
            self.problem += (
                self.w_k_r[0, r_idx] == 1,
                f"Adaptive multi-path routing constraints (16.1:{r_idx})",
            )
            self.problem += (
                self.w_k_r[max_paths[r_idx] - 1, r_idx] == 1,
                f"Adaptive multi-path routing constraints (16.2:{r_idx})",
            )
            for k in range(max_paths[r_idx] - 2):
                self.problem += (
                    self.w_k_r[(k, r_idx)] >= self.w_k_r[(k + 1, r_idx)],
                    f"Adaptive multi-path routing constraints (17:{r_idx}:{k})",
                )

        for r_idx, r in enumerate(inst.requests):
            for k in range(max_paths[r_idx]):
                self.problem += (
                    lpSum(
                        self.Phi_k_rm[(k, r_idx, m_idx)] * m.relative_bpsk_rate()
                        for m_idx, m in enumerate(ModulationFormat.all())
                    )
                    + (1 - self.w_k_r[(k, r_idx)]) * r.bpsk_fs_count
                    >= r.bpsk_fs_count
                    * lpSum(
                        self.xi_i_r[(k, r_idx)] / k for k in range(1, max_paths[r_idx])
                    ),
                    f"Modulation adaption constraints (18:{r_idx}:{k})",
                )

        for r_idx, r in enumerate(inst.requests):
            for k in range(max_paths[r_idx]):
                for a_idx, a in enumerate(arcs):
                    self.problem += (
                        self.Phi_k_ra[(k, r_idx, a_idx)]
                        <= S * self.p_k_ra[(k, r_idx, a_idx)],
                        "Adaptive multi-path routing constraints "
                        + f"(19:{r_idx}:{k}:{a_idx})",
                    )

        for r_idx, r in enumerate(inst.requests):
            for k in range(max_paths[r_idx]):
                for a_idx, _ in enumerate(arcs):
                    self.problem += (
                        self.Phi_k_ra[(k, r_idx, a_idx)] <= self.Phi_k_r[(k, r_idx)],
                        "Adaptive multi-path routing constraints "
                        + f"(20:{r_idx}:{k}:{a_idx})",
                    )

        for r_idx, r in enumerate(inst.requests):
            for k in range(max_paths[r_idx]):
                for a_idx, a in enumerate(arcs):
                    self.problem += (
                        self.Phi_k_ra[(k, r_idx, a_idx)]
                        >= self.Phi_k_r[(k, r_idx)]
                        - S * (1 - self.p_k_ra[(k, r_idx, a_idx)]),
                        "Adaptive multi-path routing constraints "
                        + f"(21:{r_idx}:{k}:{a_idx})",
                    )

        for r_idx, r in enumerate(inst.requests):
            for a_idx, _ in enumerate(arcs):
                for k in range(max_paths[r_idx]):
                    for k_prime in range(max_paths[r_idx]):
                        if k <= k_prime:
                            continue
                        self.problem += (
                            self.p_k_ra[(k, r_idx, a_idx)]
                            + self.p_k_ra[(k_prime, r_idx, a_idx)]
                            - 1
                            <= self.gamma_kkp_r[(k, k_prime, r_idx)],
                            "Spectrum allocation constraints "
                            + f"(22:{r_idx}:{k}:{k_prime}:{a_idx})",
                        )
        for r_idx, r in enumerate(inst.requests):
            for k in range(max_paths[r_idx]):
                for k_prime in range(max_paths[r_idx]):
                    if k <= k_prime:
                        continue
                    self.problem += (
                        self.gamma_kkp_r[(k, k_prime, r_idx)]
                        == self.gamma_kkp_r[(k_prime, k, r_idx)],
                        f"Spectrum allocation constraints (23:{r_idx}:{k}:{k_prime})",
                    )
        for r_idx, _ in enumerate(inst.requests):
            for r_prime_idx, _ in enumerate(inst.requests):
                if r_idx >= r_prime_idx:
                    continue
                for a_idx, _ in enumerate(arcs):
                    for k in range(max_paths[r_idx]):
                        for k_prime in range(max_paths[r_prime_idx]):
                            self.problem += (
                                self.p_k_ra[(k, r_idx, a_idx)]
                                + self.p_k_ra[(k_prime, r_prime_idx, a_idx)]
                                - 1
                                <= self.gamma_kkp_rrp[(k, k_prime, r_idx, r_prime_idx)],
                                "Spectrum allocation constraints "
                                + f"(24:{r_idx}:{r_prime_idx}:{k}:{k_prime}:{a_idx})",
                            )
        for r_idx, _ in enumerate(inst.requests):
            for r_idx_prime, _ in enumerate(inst.requests):
                if r_idx >= r_idx_prime:
                    continue
                for k in range(max_paths[r_idx]):
                    for k_prime in range(max_paths[r_idx_prime]):
                        self.problem += (
                            self.gamma_kkp_rrp[(k, k_prime, r_idx, r_idx_prime)]
                            == self.gamma_kkp_rrp[(k_prime, k, r_idx_prime, r_idx)],
                            "Spectrum allocation constraints "
                            + f"(25:{r_idx}:{r_idx_prime}:{k}:{k_prime})",
                        )

        for r_idx, r in enumerate(inst.requests):
            for k in range(max_paths[r_idx]):
                for k_prime in range(max_paths[r_idx]):
                    if k <= k_prime:
                        continue
                    self.problem += (
                        self.beta_kkp_r[(k, k_prime, r_idx)]
                        + self.beta_kkp_r[k_prime, k, r_idx]
                        == 1,
                        f"Spectrum allocation constraints (26:{r_idx}:{k}:{k_prime})",
                    )
        for r_idx, _ in enumerate(inst.requests):
            for r_prime_idx, _ in enumerate(inst.requests):
                if r_idx <= r_prime_idx:
                    continue
                for k in range(max_paths[r_idx]):
                    for k_prime in range(max_paths[r_prime_idx]):
                        self.problem += (
                            self.beta_kkp_rrp[(k, k_prime, r_idx, r_prime_idx)]
                            + self.beta_kkp_rrp[(k_prime, k, r_prime_idx, r_idx)]
                            == 1,
                            "Spectrum allocation constraints "
                            + f"(27:{r_idx}:{r_prime_idx}:{k}:{k_prime})",
                        )
        for r_idx, r in enumerate(inst.requests):
            for k in range(max_paths[r_idx]):
                self.problem += (
                    self.g_k_r[(k, r_idx)] + self.Phi_k_r[(k, r_idx)] <= self.Delta,
                    f"Spectrum allocation constraints (28:{r_idx}:{k})",
                )

        M = int(1e6)
        for r_idx, r in enumerate(inst.requests):
            for k in range(max_paths[r_idx]):
                for k_prime in range(max_paths[r_idx]):
                    if k == k_prime:
                        continue
                    self.problem += (
                        self.g_k_r[(k, r_idx)]
                        + self.Phi_k_r[(k, r_idx)]
                        - self.g_k_r[(k_prime, r_idx)]
                        <= M
                        * (
                            2
                            - self.gamma_kkp_r[(k, k_prime, r_idx)]
                            - self.beta_kkp_r[(k, k_prime, r_idx)]
                        ),
                        f"Spectrum allocation constraints (29:{r_idx}:{k}:{k_prime})",
                    )
        for r_idx, r in enumerate(inst.requests):
            for r_idx_prime, r_prime in enumerate(inst.requests):
                if r_idx == r_idx_prime:
                    continue
                for k in range(max_paths[r_idx]):
                    for k_prime in range(max_paths[r_idx_prime]):
                        self.problem += (
                            self.g_k_r[(k, r_idx)]
                            + self.Phi_k_r[(k, r_idx)]
                            - self.g_k_r[(k_prime, r_idx_prime)]
                            <= M
                            * (
                                2
                                - self.gamma_kkp_rrp[(k, k_prime, r_idx, r_idx_prime)]
                                - self.beta_kkp_rrp[(k, k_prime, r_idx, r_idx_prime)]
                            ),
                            "Spectrum allocation constraints "
                            + f"(30:{r_idx}:{r_idx_prime}:{k}:{k_prime})",
                        )

    def solve(self, msg: bool, time_limit: int | None) -> ILPCDPSolution:
        assert self.problem.objective is not None
        status = self.problem.solve(PULP_CBC_CMD(msg=msg, timeLimit=time_limit))
        objective = self.problem.objective.value()
        log.debug(SL("ILP solve status", status=status, objective=objective))

        var_values = self.problem.variablesDict()
        return ILPCDPSolution(status, var_values, objective)

    def check_solution(
        self,
        all_routes: ilist[ilist[Route]],
        start_indices: dict[int, int],
        content_placement: dict[int, set[int]],
        total_fs: int | None = None,
        mofi: int | None = None,
    ) -> None:
        values: dict[str, int] = {}

        def set_value(v: LpVariable, value: int | bool) -> None:
            values[v.name] = int(value)

        def get_value(v: LpVariable) -> int:
            if v.name not in values:
                raise ValueError(f"Variable {v.name} not set")
            return values[v.name]

        def path_idx(k: int, r_idx: int) -> int:
            if k == self.max_paths[r_idx] - 1:
                return len(all_routes[r_idx]) - 1
            elif k < len(all_routes[r_idx]) - 1:
                return k
            else:
                return len(all_routes[r_idx])

        arcs: list[tuple[int, int, int]] = [
            (u, v, self.inst.topology.graph.edges[u, v]["weight"])
            for u, v in self.inst.topology.graph.edges
        ]

        for dc in self.inst.possible_dc_positions:
            set_value(
                self.theta_d[dc],
                any(dc in dc_pos for content, dc_pos in content_placement.items()),
            )

        for r_idx, r in enumerate(self.inst.requests):
            for k in range(self.max_paths[r_idx]):
                pi = path_idx(k, r_idx)
                for a_idx, a in enumerate(arcs):
                    set_value(
                        self.p_k_ra[(k, r_idx, a_idx)],
                        len(all_routes[r_idx]) > pi
                        and all_routes[r_idx][pi].has_edge(a[:2]),
                    )
        for r_idx, r in enumerate(self.inst.requests):
            for k in range(self.max_paths[r_idx]):
                pi = path_idx(k, r_idx)
                for d in self.avail_dcs:
                    set_value(
                        self.Lambda_k_rd[(k, r_idx, d)],
                        len(all_routes[r_idx]) > pi
                        and all_routes[r_idx][pi].node_list[-1] == d,
                    )
        for r_idx, r in enumerate(self.inst.requests):
            for k in range(self.max_paths[r_idx]):
                pi = path_idx(k, r_idx)
                for z_idx, z in enumerate(self.inst.topology.dzs):
                    set_value(
                        self.alpha_k_rz[(k, r_idx, z_idx)],
                        len(all_routes[r_idx]) > pi
                        and any(
                            a[:2] in all_routes[r_idx][pi].edges()
                            and a_idx in self.dz_arcs[z_idx]
                            for a_idx, a in enumerate(self.arcs)
                        ),
                    )
        for d in self.avail_dcs:
            for cr in self.contents:
                set_value(
                    self.R_cr_d[(cr, d)],
                    any(
                        any(routes.node_list[-1] == d for routes in all_routes[r_idx])
                        for r_idx, r in enumerate(self.inst.requests)
                        if r.content_id == cr
                    ),
                )

        num_fses = tuple(
            tuple(
                ceil(
                    req.bpsk_fs_count
                    / route.format.relative_bpsk_rate()
                    / (len(routes) - 1)
                )
                for route in routes
            )
            for req, routes in zip(self.inst.requests, all_routes)
        )

        for r_idx, r in enumerate(self.inst.requests):
            for k in range(self.max_paths[r_idx]):
                pi = path_idx(k, r_idx)
                for m_idx, m in enumerate(ModulationFormat.all()):
                    set_value(
                        self.Phi_k_rm[(k, r_idx, m_idx)],
                        num_fses[r_idx][pi]
                        if pi < len(all_routes[r_idx])
                        and m == all_routes[r_idx][pi].format
                        else 0,
                    )
        for r_idx, r in enumerate(self.inst.requests):
            for k in range(self.max_paths[r_idx]):
                set_value(
                    self.w_k_r[(k, r_idx)],
                    len(all_routes[r_idx]) > path_idx(k, r_idx),
                )
        for r_idx, r in enumerate(self.inst.requests):
            for i in range(1, self.max_paths[r_idx]):
                set_value(self.xi_i_r[(i, r_idx)], len(all_routes[r_idx]) == i + 1)
        for r_idx, r in enumerate(self.inst.requests):
            for k in range(self.max_paths[r_idx]):
                pi = path_idx(k, r_idx)
                for a_idx, a in enumerate(arcs):
                    set_value(
                        self.Phi_k_ra[(k, r_idx, a_idx)],
                        num_fses[r_idx][pi]
                        if len(all_routes[r_idx]) > pi
                        and all_routes[r_idx][pi].has_edge(a[:2])
                        else 0,
                    )
        for r_idx, r in enumerate(self.inst.requests):
            for k in range(self.max_paths[r_idx]):
                pi = path_idx(k, r_idx)
                set_value(
                    self.Phi_k_r[(k, r_idx)],
                    num_fses[r_idx][pi] if len(all_routes[r_idx]) > pi else 0,
                )
        start_indices_unflattened: list[list[int]] = []
        # unflatten start_indices based on all_routes
        idx = 0
        for routes in all_routes:
            indices: list[int] = []
            for route in routes:
                indices.append(start_indices[idx])
                idx += 1
            start_indices_unflattened.append(indices)

        log.debug(
            SL(
                "Unflattened DSA results",
                num_fses=num_fses,
                start_indices=start_indices_unflattened,
            )
        )

        for r_idx, r in enumerate(self.inst.requests):
            for k in range(self.max_paths[r_idx]):
                pi = path_idx(k, r_idx)
                set_value(
                    self.g_k_r[(k, r_idx)],
                    start_indices_unflattened[r_idx][pi]
                    if len(all_routes[r_idx]) > pi
                    else 0,
                )
        for r_idx, r in enumerate(self.inst.requests):
            for k in range(self.max_paths[r_idx]):
                for k_prime in range(self.max_paths[r_idx]):
                    if k >= k_prime:
                        continue
                    pi = path_idx(k, r_idx)
                    pi_prime = path_idx(k_prime, r_idx)
                    value = 0
                    if pi < len(all_routes[r_idx]) and pi_prime < len(
                        all_routes[r_idx]
                    ):
                        value = (
                            start_indices_unflattened[r_idx][pi]
                            < start_indices_unflattened[r_idx][pi_prime]
                        )
                    set_value(
                        self.beta_kkp_r[(k, k_prime, r_idx)],
                        value,
                    )
                    set_value(
                        self.beta_kkp_r[(k_prime, k, r_idx)],
                        1 - value,
                    )
        for r_idx, r in enumerate(self.inst.requests):
            for r_prime_idx, r_prime in enumerate(self.inst.requests):
                if r_idx >= r_prime_idx:
                    continue
                for k in range(self.max_paths[r_idx]):
                    for k_prime in range(self.max_paths[r_prime_idx]):
                        pi = path_idx(k, r_idx)
                        pi_prime = path_idx(k_prime, r_prime_idx)
                        value = 0
                        if pi < len(all_routes[r_idx]) and pi_prime < len(
                            all_routes[r_prime_idx]
                        ):
                            value = (
                                start_indices_unflattened[r_idx][pi]
                                < start_indices_unflattened[r_prime_idx][pi_prime]
                            )
                        set_value(
                            self.beta_kkp_rrp[(k, k_prime, r_idx, r_prime_idx)],
                            value,
                        )
                        set_value(
                            self.beta_kkp_rrp[(k_prime, k, r_prime_idx, r_idx)],
                            1 - value,
                        )
        for r_idx, r in enumerate(self.inst.requests):
            for k in range(self.max_paths[r_idx]):
                for k_prime in range(self.max_paths[r_idx]):
                    if k >= k_prime:
                        continue
                    pi = path_idx(k, r_idx)
                    pi_prime = path_idx(k_prime, r_idx)
                    value = 0
                    if pi < len(all_routes[r_idx]) and pi_prime < len(
                        all_routes[r_idx]
                    ):
                        value = any(
                            all_routes[r_idx][pi_prime].has_edge(edge)
                            for edge in all_routes[r_idx][pi].edges()
                        )
                    set_value(
                        self.gamma_kkp_r[(k, k_prime, r_idx)],
                        value,
                    )
                    set_value(
                        self.gamma_kkp_r[(k_prime, k, r_idx)],
                        value,
                    )
        for r_idx, r in enumerate(self.inst.requests):
            for r_prime_idx, r_prime in enumerate(self.inst.requests):
                if r_idx >= r_prime_idx:
                    continue
                for k in range(self.max_paths[r_idx]):
                    for k_prime in range(self.max_paths[r_prime_idx]):
                        pi = path_idx(k, r_idx)
                        pi_prime = path_idx(k_prime, r_prime_idx)
                        value = 0
                        if pi < len(all_routes[r_idx]) and pi_prime < len(
                            all_routes[r_prime_idx]
                        ):
                            value = any(
                                all_routes[r_idx][pi].has_edge(edge)
                                for edge in all_routes[r_prime_idx][pi_prime].edges()
                            )
                        set_value(
                            self.gamma_kkp_rrp[(k, k_prime, r_idx, r_prime_idx)],
                            value,
                        )
                        set_value(
                            self.gamma_kkp_rrp[(k_prime, k, r_prime_idx, r_idx)],
                            value,
                        )
        for r_idx, r in enumerate(self.inst.requests):
            for k in range(self.max_paths[r_idx]):
                pi = path_idx(k, r_idx)
                for m_idx, m in enumerate(ModulationFormat.all()):
                    set_value(
                        self.b_k_mr[(k, m_idx, r_idx)],
                        len(all_routes[r_idx]) > pi
                        and all_routes[r_idx][pi].format == m,
                    )

        calc_mofi = max(
            a + b
            for sis, num_fs in zip(start_indices_unflattened, num_fses)
            for a, b in zip(sis, num_fs)
        )

        set_value(self.Delta, calc_mofi)

        def calc_expr(expr: LpAffineExpression | LpVariable) -> float:
            if isinstance(expr, LpVariable):
                return get_value(expr)
            else:
                return cast(
                    float,
                    sum(get_value(v) * coeff for v, coeff in expr.items())
                    + expr.constant,
                )

        def calc_constraint(
            expr: LpConstraint,
        ) -> tuple[float, dict[str, dict[str, float]]]:
            sum = expr.constant
            var_values: dict[str, dict[str, float]] = {
                "CONST": {
                    "coeff": 1,
                    "value": expr.constant,
                },
            }
            for v, x in expr.items():
                value = get_value(v)
                var_values[v.name] = {
                    "coeff": x,
                    "value": value,
                }
                sum += value * x
            return sum, var_values

        assert mofi is None or abs(calc_expr(self.Delta) - mofi) < 1e-4
        assert total_fs is None or abs(calc_expr(self.total_fs) - total_fs) < 1e-4

        # verify bounds
        num_true = 0
        num_false = 0
        num_incomplete = 0
        for variable in self.problem.variables():
            try:
                value = get_value(variable)
                if (variable.lowBound is not None and value < variable.lowBound) or (
                    variable.upBound is not None and value > variable.upBound
                ):
                    num_false += 1
                    log.debug(
                        SL(
                            "Variable out of bounds",
                            name=variable.name,
                            value=value,
                            lowBound=variable.lowBound,
                            upBound=variable.upBound,
                        ),
                    )
                    continue
                if variable.cat == LpBinary and value not in (0, 1):
                    num_false += 1
                    log.debug(
                        SL(
                            "Variable is not binary",
                            name=variable.name,
                            value=value,
                        )
                    )
                    continue
                if variable.cat == LpInteger and not isinstance(value, int):
                    num_false += 1
                    log.debug(
                        SL(
                            "Variable is not integer",
                            name=variable.name,
                            value=value,
                        )
                    )
                    continue
                num_true += 1
            except ValueError:
                log.debug(SL("Incomplete variable", name=variable.name))
                num_incomplete += 1

        log.info(
            SL(
                "Variable verification results",
                num_vars=len(self.problem.variables()),
                num_true=num_true,
                num_false=num_false,
                num_incomplete=num_incomplete,
            )
        )
        assert num_false == 0, "Some variables are out of bounds or not binary/integer"

        # verify constraints
        num_true = 0
        num_false = 0
        num_incomplete = 0
        for name, constraint in self.problem.constraints.items():
            try:
                expr_value, var_values = calc_constraint(constraint)
                if abs(expr_value) < 1e-4:
                    cmp_result = 0
                else:
                    cmp_result = 1 if expr_value > 0 else -1
                cmp_result *= constraint.sense
                if cmp_result < 0:
                    num_false += 1
                    log.debug(
                        SL(
                            "Constraint is violated",
                            name=name,
                            value=expr_value,
                            sense=constraint.sense,
                            var_values=var_values,
                        )
                    )
                else:
                    num_true += 1
            except ValueError:
                num_incomplete += 1

        log.info(
            SL(
                "Constraint verification results",
                num_vars=len(self.problem.variables()),
                num_true=num_true,
                num_false=num_false,
                num_incomplete=num_incomplete,
            )
        )
        assert num_false == 0, "Some constraints are violated"
