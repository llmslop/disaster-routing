from functools import cache
import logging
from math import isnan
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
from ..random.random import Random
from ..random.config import RandomConfig
from .routing_algo import InfeasibleRouteError, Route, RoutingAlgorithm

log = logging.getLogger(__name__)


def randomized_dfs(
    random: Random,
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
    random.stdlib.shuffle(neighbors)

    for neighbor in neighbors:
        if neighbor not in visited:
            result = randomized_dfs(random, graph, neighbor, end, visited, path)
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

    def fitness(self, evaluator: FitnessEvaluator) -> float:
        if isnan(self.fitness_value):
            self.fitness_value = evaluator.evaluate(self)
        return self.fitness_value

    @staticmethod
    def random(
        random: Random, inst: Instance, content_placement: dict[int, list[int]]
    ) -> "Individual":
        all_routes: list[ilist[Route]] = [() for _ in inst.requests]
        num_retries = 0
        for k, req in enumerate(inst.requests):
            while True:
                all_routes[k] = ()
                while True:
                    route = Individual.generate_new_route(
                        random, inst, req, all_routes[k], content_placement
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
        random: Random,
        inst: Instance,
        content_placement: dict[int, list[int]],
        other: "Individual",
        num_retries_per_req: int,
    ) -> "Individual | None":
        all_routes: list[ilist[Route]] = []
        for i, (r1, r2) in enumerate(zip(self.all_routes, other.all_routes)):
            all_routes.append(random.stdlib.choice([r1, r2]))
        return Individual(tuple(all_routes))

    @staticmethod
    def generate_new_route(
        random: Random,
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
            route = randomized_dfs(random, topology.graph, req.source, dc)
            if route is None:
                continue
            try:
                return Route(inst.topology, ilist[int](route))
            except InfeasibleRouteError:
                pass
        return None

    def mutate(
        self,
        random: Random,
        inst: Instance,
        content_placement: dict[int, list[int]],
        mut_rate: float,
        num_retries_per_req: int,
    ) -> "Individual | None":
        try:
            all_routes: list[ilist[Route]] = list(self.all_routes)
            k = min(random.numpy.geometric(0.8), len(all_routes) - 1)
            ks = random.stdlib.choices(range(len(all_routes)), k=k)

            for k in ks:
                chances = {
                    "new": 1,
                    "reroute": 1,
                    "prune": 1,
                }

                if len(all_routes[k]) <= 2:
                    del chances["prune"]

                # try to generate new route
                new_route = Individual.generate_new_route(
                    random, inst, inst.requests[k], all_routes[k], content_placement
                )
                if new_route is None:
                    del chances["new"]

                choices = list(chances.keys())
                chances = [float(chances[c]) for c in choices]
                match random.stdlib.choices(choices, weights=chances)[0]:
                    case "new":
                        assert new_route is not None
                        all_routes[k] = all_routes[k] + (new_route,)
                    case "reroute":
                        routes: ilist[Route] = ()
                        for _ in range(num_retries_per_req):
                            while True:
                                route = Individual.generate_new_route(
                                    random,
                                    inst,
                                    inst.requests[k],
                                    routes,
                                    content_placement,
                                )
                                if route is None or random.stdlib.random() < 0.25:
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
                        _ = route_list.pop(
                            random.stdlib.randint(0, len(route_list) - 1)
                        )
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
    random: Random
    pop_size: int
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
        random: Random,
        pop_size: int,
        cr_rate: float,
        mut_rate: float,
        cr_num_retries_per_req: int,
        mut_num_retries_per_req: int,
        elitism_rate: float,
    ):
        self.inst = inst
        self.content_placement = content_placement
        self.evaluator = evaluator
        self.random = random
        self.pop_size = pop_size
        self.cr_rate = cr_rate
        self.mut_rate = mut_rate
        self.cr_num_retries_per_req = cr_num_retries_per_req
        self.mut_num_retries_per_req = mut_num_retries_per_req
        self.elitism_rate = elitism_rate
        log.debug("Generating initial population...")
        self.population: list[Individual] = [
            Individual.random(self.random, inst, content_placement)
            for _ in range(pop_size)
        ]

    def select(self) -> list[Individual]:
        # Tournament selection
        tournament_size = 3
        selected: list[Individual] = []
        for _ in range(self.pop_size):
            competitors = self.random.stdlib.sample(self.population, tournament_size)
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
                parent1 = self.random.stdlib.choice(selected)
                parent2 = self.random.stdlib.choice(selected)
                child: Individual | None = None
                if self.random.stdlib.random() < self.cr_rate:
                    child = parent1.crossover(
                        self.random,
                        self.inst,
                        self.content_placement,
                        parent2,
                        self.cr_num_retries_per_req,
                    )
                    cr_success_rate.update(child is not None)
                if child is None:
                    child = parent1

                if self.random.stdlib.random() < self.mut_rate:
                    child = child.mutate(
                        self.random,
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
    cr_rate: float
    mut_rate: float
    cr_num_retries_per_req: int
    mut_num_retries_per_req: int
    elitism_rate: float
    random: Random

    def __init__(
        self,
        evaluator: Evaluator,
        dsa_solver: DSASolverConfig,
        num_gens: int,
        pop_size: int,
        cr_rate: float,
        mut_rate: float,
        cr_num_retries_per_req: int,
        mut_num_retries_per_req: int,
        elitism_rate: float,
        random: RandomConfig,
        **_: object,
    ):
        self.dsa_solver = dsa_solver
        self.evaluator = evaluator
        self.num_gens = num_gens
        self.pop_size = pop_size
        self.cr_rate = cr_rate
        self.mut_rate = mut_rate
        self.cr_num_retries_per_req = cr_num_retries_per_req
        self.mut_num_retries_per_req = mut_num_retries_per_req
        self.elitism_rate = elitism_rate
        self.random = instantiate(random)

    @override
    def route_instance(
        self, inst: Instance, content_placement: dict[int, list[int]]
    ) -> ilist[ilist[Route]]:
        sga = SGA(
            inst,
            FitnessEvaluator(inst, self.evaluator, self.dsa_solver),
            content_placement,
            self.random,
            self.pop_size,
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
