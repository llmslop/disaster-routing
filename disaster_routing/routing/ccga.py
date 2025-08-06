from typing import override

from ..instances.request import Request
from ..instances.instance import Instance
from ..topologies.topology import Topology
from ..utils.ilist import ilist
from ..random.random import Random
from ..conflicts.solver import DSASolver
from ..eval.evaluator import Evaluator
from .routing_algo import Route, RoutingAlgorithm


class CCGARoutingAlgorithm(RoutingAlgorithm):
    dsa_solver: DSASolver
    evaluator: Evaluator

    num_gens: int
    pop_size: int
    cr_rate: float
    mut_rate: float
    cr_num_retries_per_req: int
    mut_num_retries_per_req: int
    random: Random

    def __init__(
        self,
        dsa_solver: DSASolver,
        evaluator: Evaluator,
        random: Random,
        num_gens: int = 100,
        pop_size: int = 10,
        cr_rate: float = 0.7,
        mut_rate: float = 0.1,
        cr_num_retries_per_req: int = 10,
        mut_num_retries_per_req: int = 10,
    ):
        self.dsa_solver = dsa_solver
        self.evaluator = evaluator
        self.random = random
        self.num_gens = num_gens
        self.pop_size = pop_size
        self.cr_rate = cr_rate
        self.mut_rate = mut_rate
        self.cr_num_retries_per_req = cr_num_retries_per_req
        self.mut_num_retries_per_req = mut_num_retries_per_req

    @override
    def route_instance(
        self, inst: Instance, content_placement: dict[int, set[int]]
    ) -> ilist[ilist[Route]]:
        return super().route_instance(inst, content_placement)

    @override
    def route_request(self, req: Request, top: Topology, dst: set[int]) -> ilist[Route]:
        raise NotImplementedError
