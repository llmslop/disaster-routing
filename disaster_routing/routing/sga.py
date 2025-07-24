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
from ..topologies.graphs import Graph
from ..utils.success_stats import SuccessRateStats
from ..utils.structlog import SL
from ..utils.ilist import ilist
from .routing_algo import InfeasibleRouteError, Route, RoutingAlgorithm
from .ndt import NodeDepthTree
from .flow import FlowRoutingAlgorithm, reconstruct_min_hop_path

log = logging.getLogger(__name__)


def randomized_dfs(
    graph: Graph,
    start: int,
    end: int,
    visited: set[int] | None = None,
    path: list[int] | None = None,
) -> list[int] | None:
    if visited is None:
        visited = set()
    if path is None:
        path = []

    visited.add(start)
    path.append(start)

    if start == end:
        return path.copy()

    neighbors = list(graph.adj[start])
    random.shuffle(neighbors)

    for neighbor in neighbors:
        if neighbor not in visited:
            result = randomized_dfs(graph, neighbor, end, visited, path)
            if result is not None:
                return result

    _ = path.pop()
    visited.remove(start)
    return None


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
    def clone_instance(inst: Instance) -> Instance:
        inst = inst.copy()
        nodes = list(inst.topology.graph.nodes)
        for u, v in zip(nodes, nodes):
            if u >= v:
                continue
            weight = inst.topology.graph.edges[u, v]["weight"]
            weight = random.random() * weight
            inst.topology.graph.edges[u, v]["weight"] = weight
            inst.topology.graph.edges[v, u]["weight"] = weight
        return inst

    @staticmethod
    def random(inst: Instance, content_placement: dict[int, list[int]]) -> "Individual":
        all_routes: list[ilist[Route]] = [() for _ in inst.requests]
        num_retries = 0
        for k, req in enumerate(inst.requests):
            while True:
                all_routes[k] = ()
                while True:
                    route = Individual.generate_new_route(
                        inst, req, all_routes[k], content_placement
                    )
                    num_retries += 1
                    if route is None:
                        break
                    all_routes[k] = all_routes[k] + (route,)
                if len(all_routes[k]) >= 2:
                    break
        log.debug(SL("Finished generating random individual", num_retries=num_retries))
        return Individual(tuple(all_routes))

    def crossover(
        self,
        inst: Instance,
        content_placement: dict[int, list[int]],
        other: "Individual",
        num_retries_per_req: int,
    ) -> "Individual | None":
        all_routes: list[ilist[Route]] = []
        for i, (r1, r2) in enumerate(zip(self.all_routes, other.all_routes)):
            all_routes.append(random.choice([r1, r2]))
        return Individual(tuple(all_routes))

    @staticmethod
    def generate_new_route(
        inst: Instance,
        req: Request,
        routes: ilist[Route],
        content_placement: dict[int, list[int]],
    ) -> Route | None:
        topology = inst.topology.copy()
        for dz in topology.dzs:
            if (
                any(dz.affects_path(route.node_list) for route in routes)
                and req.source not in dz.nodes
            ):
                dz.remove_from_graph(topology.graph)
        for dc in content_placement[req.content_id]:
            route = randomized_dfs(topology.graph, req.source, dc)
            if route is None:
                continue
            try:
                return Route(inst.topology, ilist[int](route))
            except InfeasibleRouteError:
                pass
        return None

    def mutate(
        self,
        inst: Instance,
        content_placement: dict[int, list[int]],
        mut_rate: float,
        num_retries_per_req: int,
    ) -> "Individual | None":
        try:
            all_routes: list[ilist[Route]] = list(self.all_routes)
            k = random.choice(range(len(all_routes)))

            chances = {
                "new": 1,
                "reroute": 1,
                "prune": 1,
            }

            if len(all_routes[k]) <= 2:
                del chances["prune"]

            # try to generate new route
            new_route = Individual.generate_new_route(
                inst, inst.requests[k], all_routes[k], content_placement
            )
            if new_route is None:
                del chances["new"]

            choices = list(chances.keys())
            chances = [float(chances[c]) for c in choices]
            match random.choices(choices, weights=chances)[0]:
                case "new":
                    assert new_route is not None
                    all_routes[k] = all_routes[k] + (new_route,)
                case "reroute":
                    routes: ilist[Route] = ()
                    for _ in range(num_retries_per_req):
                        while True:
                            route = Individual.generate_new_route(
                                inst, inst.requests[k], routes, content_placement
                            )
                            if route is None or random.random() < 0.25:
                                break
                            routes = routes + (route,)
                        if len(routes) < 2:
                            routes = ()
                            continue
                    if len(routes) < 2:
                        return None
                    all_routes[k] = routes
                case "prune":
                    route_list = list(all_routes[k])
                    _ = route_list.pop(random.randint(0, len(route_list) - 1))
                    all_routes[k] = ilist[Route](route_list)
                    assert len(all_routes[k]) >= 2
                case _:
                    return None

            return Individual(tuple(all_routes))
        except InfeasibleRouteError:
            return None


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
    elitism_rate: float

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
        elitism_rate: float,
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
        self.elitism_rate = elitism_rate
        log.debug("Generating initial population...")
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
        cr_success_rate = SuccessRateStats()
        mut_success_rate = SuccessRateStats()
        for gen in range(generations):
            selected = self.select()
            next_population: list[Individual] = []

            # elitism selection
            elitism_count = max(
                1, int(self.elitism_rate * self.pop_size)
            )  # configurable rate, at least 1
            next_population.extend(
                sorted(
                    self.population,
                    key=lambda ind: ind.fitness(self.evaluator),
                )[:elitism_count]
            )

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
                    cr_success_rate.update(child is not None)
                if child is None:
                    child = parent1

                if random.random() < self.mut_rate:
                    child = child.mutate(
                        self.inst,
                        self.content_placement,
                        self.mut_rate,
                        self.mut_num_retries_per_req,
                    )
                    mut_success_rate.update(child is not None)
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
                    "Fitness eval count",
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
    elitism_rate: float

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
        elitism_rate: float,
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
        self.elitism_rate = elitism_rate

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
            self.elitism_rate,
        )
        sga.evolve(self.num_gens)
        return sga.best().all_routes

    @override
    def route_request(
        self, req: Request, top: Topology, dst: list[int]
    ) -> ilist[Route]:
        raise NotImplementedError()
