from functools import cache
import logging
from math import isnan
import random
from typing import Callable, cast, override

from hydra.utils import instantiate

from ..conflicts.config import DSASolverConfig
from ..conflicts.conflict_graph import ConflictGraph
from ..conflicts.solver import DSASolver
from ..eval.evaluator import Evaluator
from ..instances.instance import Instance
from ..instances.request import Request
from ..topologies.topology import Topology
from ..utils.welford import RunningStats
from ..utils.structlog import SL
from ..utils.ilist import ilist
from .routing_algo import Route, RoutingAlgorithm
from .ndt import NodeDepthTree
from .flow import reconstruct_min_hop_path
from .greedy import GreedyRoutingAlgorithm

log = logging.getLogger(__name__)


class FitnessEvaluator:
    inst: Instance
    eval: Evaluator
    dsa_config: DSASolverConfig
    eval_cached: Callable[[ilist[ilist[Route]]], float]
    fitness_eval_count: int = 0

    def eval_tuple(self, routes: ilist[ilist[Route]]) -> float:
        conflict_graph = ConflictGraph(self.inst, routes)
        dsa_solver = cast(DSASolver, instantiate(self.dsa_config, conflict_graph))
        _, mofi = dsa_solver.solve()
        self.fitness_eval_count = self.fitness_eval_count + 1
        result = self.eval.evaluate(conflict_graph.total_fs(), mofi)
        log.debug(
            SL(
                "SGA fitness",
                routes=[[r.node_list for r in rs] for rs in routes],
                result=result,
            )
        )
        return result

    def __init__(self, inst: Instance, eval: Evaluator, dsa_config: DSASolverConfig):
        self.inst = inst
        self.eval = eval
        self.dsa_config = dsa_config

        self.eval_cached = cache(lambda inp: self.eval_tuple(inp))

    def evaluate(self, indv: "Individual") -> float:
        return self.eval_cached(indv.all_routes)


class Individual:
    all_routes: ilist[ilist[Route]]
    fitness_value: float = float("nan")

    def __init__(self, all_routes: ilist[ilist[Route]]):
        assert all(len(routes) > 1 for routes in all_routes)
        self.all_routes = tuple(
            tuple(sorted(routes, key=lambda r: r.node_list)) for routes in all_routes
        )

    def to_ndts(self) -> list[NodeDepthTree]:
        return [NodeDepthTree.from_routes(routes) for routes in self.all_routes]

    @staticmethod
    def from_ndts(
        inst: Instance,
        content_placement: dict[int, list[int]],
        ndts: list[NodeDepthTree],
    ) -> "Individual | None":
        all_routes: list[ilist[Route]] = []
        for req, ndt in zip(inst.requests, ndts):
            indv = Individual.route_from_ndt(
                inst.topology, req, content_placement[req.content_id], ndt
            )
            if indv is None:
                return None
            all_routes.append(indv)
        return Individual(tuple(all_routes))

    @staticmethod
    def route_from_ndt(
        top: Topology, req: Request, dst: list[int], ndt: NodeDepthTree
    ) -> ilist[Route] | None:
        if req.source in dst:
            return (Route(top, (req.source,)), Route(top, (req.source,)))
        layers: list[list[int]] = []
        src_dz = [dz_i for dz_i, dz in enumerate(top.dzs) if req.source in dz.nodes][0]
        dst_dzs = {
            dz_i
            for dz_i, dz in enumerate(top.dzs)
            if any(dst_node in dz.nodes for dst_node in dst)
        }

        def put_layer(idx: int, elem: int):
            nonlocal layers
            layers += [[] for _ in range(max(0, idx + 1 - len(layers)))]
            layers[idx].append(elem)

        for dz_idx, depth in enumerate(ndt.depths):
            if depth >= 0:
                put_layer(depth, dz_idx)

        if len(layers[0]) != 1:
            return None

        route_dz_lists: dict[int, set[ilist[int]]] = {src_dz: {(src_dz,)}}

        def update_route(prev_dz: int, next_dz: int):
            old_routes = route_dz_lists[prev_dz]
            old_route = random.choice(list(old_routes))
            new_route: tuple[int, ...] = old_route + (next_dz,)
            if any(req.source not in top.dzs[dz].nodes for dz in old_route):
                old_routes.remove(old_route)
            if next_dz not in route_dz_lists:
                route_dz_lists[next_dz] = {new_route}
            else:
                route_dz_lists[next_dz].add(new_route)

        def connected(dzi: int, dzj: int) -> bool:
            return any(
                top.graph.has_edge(ni, nj)
                for ni in top.dzs[dzi].nodes
                for nj in top.dzs[dzj].nodes
            )

        for i in range(len(layers) - 1):
            prev_layer, cur_layer = layers[i], layers[i + 1]

            random.shuffle(cur_layer)
            for dz in cur_layer:
                prev_dzs = [
                    prev_dz
                    for prev_dz in prev_layer
                    if connected(prev_dz, dz)
                    and len(route_dz_lists.get(prev_dz, [])) > 0
                ]
                if len(prev_dzs) == 0:
                    continue
                prev_dz = random.choice(prev_dzs)
                update_route(prev_dz, dz)

        valid_routes: list[Route] = []
        for _, routes in route_dz_lists.items():
            for route in routes:
                if route[-1] in dst_dzs:
                    group_path = [top.dzs[dz].nodes for dz in route]
                    route = reconstruct_min_hop_path(top.graph, group_path)
                    if len(route) > 0:
                        valid_routes.append(Route(top, route))

        if len(valid_routes) <= 1:
            return None

        return tuple(valid_routes)

    def fitness(self, evaluator: FitnessEvaluator) -> float:
        if isnan(self.fitness_value):
            self.fitness_value = evaluator.evaluate(self)
        return self.fitness_value

    @staticmethod
    def random(inst: Instance, content_placement: dict[int, list[int]]) -> "Individual":
        # TODO: properly initialize initial population
        base = Individual(
            GreedyRoutingAlgorithm().route_instance(inst, content_placement)
        )
        mut = base.mutate(inst, content_placement, 0.5, 10)
        return base if mut is None else mut

    def crossover(
        self,
        inst: Instance,
        content_placement: dict[int, list[int]],
        other: "Individual",
        num_retries_per_req: int,
    ) -> "Individual | None":
        all_routes: list[ilist[Route]] = []
        for req, ndt1, ndt2 in zip(inst.requests, self.to_ndts(), other.to_ndts()):
            success = False
            for _ in range(num_retries_per_req):
                ndt = ndt1.crossover(ndt2)
                route = Individual.route_from_ndt(
                    inst.topology, req, content_placement[req.content_id], ndt
                )
                if route is not None:
                    all_routes.append(route)
                    success = True
                    break
            if not success:
                return None
        return Individual(tuple(all_routes))

    def mutate(
        self,
        inst: Instance,
        content_placement: dict[int, list[int]],
        mut_rate: float,
        num_retries_per_req: int,
    ) -> "Individual | None":
        all_routes: list[ilist[Route]] = []
        for req, ndt1 in zip(inst.requests, self.to_ndts()):
            success = False
            for _ in range(num_retries_per_req):
                ndt = ndt1.mutate(mut_rate)
                indv = Individual.route_from_ndt(
                    inst.topology, req, content_placement[req.content_id], ndt
                )
                if indv is not None:
                    all_routes.append(indv)
                    success = True
                    break
            if not success:
                return None
        return Individual(tuple(all_routes))


class SGA:
    inst: Instance
    content_placement: dict[int, list[int]]
    evaluator: FitnessEvaluator
    pop_size: int
    max_depth: int
    cr_rate: float
    mut_rate: float
    cr_num_retries_per_req: int
    mut_num_retries_per_req: int

    def __init__(
        self,
        inst: Instance,
        evaluator: FitnessEvaluator,
        content_placement: dict[int, list[int]],
        pop_size: int,
        max_depth: int,
        cr_rate: float,
        mut_rate: float,
        cr_num_retries_per_req: int,
        mut_num_retries_per_req: int,
    ):
        self.inst = inst
        self.content_placement = content_placement
        self.evaluator = evaluator
        self.pop_size = pop_size
        self.max_depth = max_depth
        self.cr_rate = cr_rate
        self.mut_rate = mut_rate
        self.cr_num_retries_per_req = cr_num_retries_per_req
        self.mut_num_retries_per_req = mut_num_retries_per_req
        self.population: list[Individual] = [
            Individual.random(inst, content_placement) for _ in range(pop_size)
        ]

    def select(self) -> list[Individual]:
        # Tournament selection
        tournament_size = 3
        selected: list[Individual] = []
        for _ in range(self.pop_size):
            competitors = random.sample(self.population, tournament_size)
            winner = self.best(competitors)
            selected.append(winner)
        return selected

    def evolve(self, generations: int) -> None:
        cr_success_rate = RunningStats()
        mut_success_rate = RunningStats()
        for gen in range(generations):
            selected = self.select()
            next_population: list[Individual] = []
            while len(next_population) < self.pop_size:
                parent1 = random.choice(selected)
                parent2 = random.choice(selected)
                child: Individual | None = None
                if random.random() < self.cr_rate:
                    child = parent1.crossover(
                        self.inst,
                        self.content_placement,
                        parent2,
                        self.cr_num_retries_per_req,
                    )
                    cr_success_rate.update(0.0 if child is None else 1.0)
                if child is None:
                    child = parent1

                child = child.mutate(
                    self.inst,
                    self.content_placement,
                    self.mut_rate,
                    self.mut_num_retries_per_req,
                )
                mut_success_rate.update(0.0 if child is None else 1.0)
                if child is None:
                    child = parent1
                next_population.append(child)

            log.debug(
                SL("Crossover success rate", gen=gen, dist=cr_success_rate.info())
            )
            log.debug(
                SL("Mutation success rate", gen=gen, dist=mut_success_rate.info())
            )
            log.debug(
                SL(
                    "Fintess eval count",
                    gen=gen,
                    count=self.evaluator.fitness_eval_count,
                )
            )
            log.debug(
                SL(
                    "Best individual",
                    gen=gen,
                    indv=[
                        [r.node_list for r in routes]
                        for routes in self.best().all_routes
                    ],
                    fitness=self.best().fitness(self.evaluator),
                )
            )
            self.population = next_population

    def best(self, collection: list[Individual] | None = None) -> Individual:
        collection = collection if collection is not None else self.population
        return min(collection, key=lambda ind: ind.fitness(self.evaluator))


class SGARoutingAlgorithm(RoutingAlgorithm):
    dsa_solver: DSASolverConfig
    evaluator: Evaluator

    num_gens: int
    pop_size: int
    max_depth: int
    cr_rate: float
    mut_rate: float
    cr_num_retries_per_req: int
    mut_num_retries_per_req: int

    def __init__(
        self,
        evaluator: Evaluator,
        dsa_solver: DSASolverConfig,
        num_gens: int,
        pop_size: int,
        max_depth: int,
        cr_rate: float,
        mut_rate: float,
        cr_num_retries_per_req: int,
        mut_num_retries_per_req: int,
        **_: object,
    ):
        self.dsa_solver = dsa_solver
        self.evaluator = evaluator
        self.num_gens = num_gens
        self.pop_size = pop_size
        self.max_depth = max_depth
        self.cr_rate = cr_rate
        self.mut_rate = mut_rate
        self.cr_num_retries_per_req = cr_num_retries_per_req
        self.mut_num_retries_per_req = mut_num_retries_per_req

    @override
    def route_instance(
        self, inst: Instance, content_placement: dict[int, list[int]]
    ) -> ilist[ilist[Route]]:
        sga = SGA(
            inst,
            FitnessEvaluator(inst, self.evaluator, self.dsa_solver),
            content_placement,
            self.pop_size,
            self.max_depth,
            self.cr_rate,
            self.mut_rate,
            self.cr_num_retries_per_req,
            self.mut_num_retries_per_req,
        )
        sga.evolve(self.num_gens)
        return sga.best().all_routes

    @override
    def route_request(
        self, req: Request, top: Topology, dst: list[int]
    ) -> ilist[Route]:
        raise NotImplementedError()
