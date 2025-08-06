#!/bin/env python3
from functools import cache
import logging
from math import isnan
from typing import Callable, override

from disaster_routing.routing.genpath import PathGenerator

from ..conflicts.conflict_graph import ConflictGraph
from ..conflicts.solver import DSASolver
from ..eval.evaluator import Evaluator
from ..instances.instance import Instance
from ..instances.request import Request
from ..topologies.topology import Topology
from ..utils.success_stats import SuccessRateStats
from ..utils.structlog import SL
from ..utils.ilist import ilist
from ..random.random import Random
from .routing_algo import InfeasibleRouteError, Route, RoutingAlgorithm

log = logging.getLogger(__name__)


class FitnessEvaluator:
    inst: Instance
    eval: Evaluator
    dsa_solver: DSASolver
    eval_cached: Callable[[ilist[ilist[Route]]], float]
    fitness_eval_count: int = 0

    def eval_tuple(self, routes: ilist[ilist[Route]]) -> float:
        conflict_graph = ConflictGraph(self.inst, routes)
        _, mofi = self.dsa_solver.solve(conflict_graph)
        self.fitness_eval_count = self.fitness_eval_count + 1
        result = self.eval.evaluate(conflict_graph.total_fs(), mofi)
        return result

    def __init__(self, inst: Instance, eval: Evaluator, dsa_solver: DSASolver):
        self.inst = inst
        self.eval = eval
        self.dsa_solver = dsa_solver

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
        random: Random,
        inst: Instance,
        pathgen: PathGenerator,
    ) -> "Individual":
        all_routes: list[ilist[Route]] = [() for _ in inst.requests]
        total_retries = 0
        for k, req in enumerate(inst.requests):
            routes, num_retries = pathgen.generate_request_routes(random, req)
            assert routes is not None
            all_routes[k] = routes
            total_retries += num_retries
        log.debug(
            SL("Finished generating random individual", total_retries=total_retries)
        )
        return Individual(tuple(all_routes))

    def crossover(
        self,
        random: Random,
        inst: Instance,
        content_placement: dict[int, set[int]],
        other: "Individual",
        num_retries_per_req: int,
    ) -> "Individual | None":
        all_routes: list[ilist[Route]] = []
        for i, (r1, r2) in enumerate(zip(self.all_routes, other.all_routes)):
            all_routes.append(random.stdlib.choice([r1, r2]))
        return Individual(tuple(all_routes))

    def mutate(
        self,
        random: Random,
        inst: Instance,
        pathgen: PathGenerator,
        content_placement: dict[int, set[int]],
        num_retries_per_req: int,
    ) -> "Individual | None":
        try:
            all_routes: list[ilist[Route]] = list(self.all_routes)
            k = min(random.numpy.geometric(0.5), len(all_routes) - 1)
            ks = random.stdlib.choices(range(len(all_routes)), k=k)

            for k in ks:
                chances = {
                    "new": 1,
                    "reroute": 1,
                    "prune": 1,
                }

                if len(all_routes[k]) <= 2:
                    del chances["prune"]

                dcs = content_placement[inst.requests[k].content_id].difference(
                    route.node_list[-1] for route in all_routes[k]
                )

                # try to generate new route
                new_route = pathgen.generate_request_route(
                    random, inst.requests[k], all_routes[k], dcs
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
                        routes, _ = pathgen.generate_request_routes(
                            random, inst.requests[k], num_retries_per_req
                        )
                        if routes is not None:
                            all_routes[k] = routes
                    case "prune":
                        route_list = list(all_routes[k])
                        _ = route_list.pop(
                            random.stdlib.randint(0, len(route_list) - 1)
                        )
                        all_routes[k] = ilist[Route](route_list)
                        assert len(all_routes[k]) >= 2
                    case _:
                        continue

            return Individual(tuple(all_routes))
        except InfeasibleRouteError:
            return None


class SGARoutingAlgorithm(RoutingAlgorithm):
    evaluator: Evaluator
    random: Random
    dsa_solver: DSASolver
    num_gens: int
    pop_size: int
    cr_rate: float
    mut_rate: float
    cr_num_retries_per_req: int
    mut_num_retries_per_req: int
    elitism_rate: float

    def __init__(
        self,
        evaluator: Evaluator,
        random: Random,
        dsa_solver: DSASolver,
        num_gens: int,
        pop_size: int,
        cr_rate: float,
        mut_rate: float,
        cr_num_retries_per_req: int,
        mut_num_retries_per_req: int,
        elitism_rate: float,
        **kwargs: object,
    ):
        self.evaluator = evaluator
        self.random = random
        self.dsa_solver = dsa_solver
        self.num_gens = num_gens
        self.pop_size = pop_size
        self.cr_rate = cr_rate
        self.mut_rate = mut_rate
        self.cr_num_retries_per_req = cr_num_retries_per_req
        self.mut_num_retries_per_req = mut_num_retries_per_req
        self.elitism_rate = elitism_rate

    def select(
        self, population: list[Individual], fitness_evaluator: FitnessEvaluator
    ) -> list[Individual]:
        # Tournament selection
        tournament_size = 3
        selected: list[Individual] = []
        for _ in range(self.pop_size):
            competitors = self.random.stdlib.sample(population, tournament_size)
            winner = self.best(competitors, fitness_evaluator)
            selected.append(winner)
        return selected

    def evolve(
        self, inst: Instance, content_placement: dict[int, set[int]], generations: int
    ) -> tuple[list[Individual], FitnessEvaluator]:
        log.debug("Generating initial population...")
        fitness_evaluator = FitnessEvaluator(inst, self.evaluator, self.dsa_solver)
        pathgen = PathGenerator(inst.topology, content_placement)
        population: list[Individual] = [
            Individual.random(self.random, inst, pathgen) for _ in range(self.pop_size)
        ]
        cr_success_rate = SuccessRateStats()
        mut_success_rate = SuccessRateStats()
        for gen in range(generations):
            selected = self.select(population, fitness_evaluator)
            next_population: list[Individual] = []

            # elitism selection
            elitism_count = max(
                1, int(self.elitism_rate * self.pop_size)
            )  # configurable rate, at least 1
            next_population.extend(
                sorted(
                    population,
                    key=lambda ind: ind.fitness(fitness_evaluator),
                )[:elitism_count]
            )

            while len(next_population) < self.pop_size:
                parent1 = self.random.stdlib.choice(selected)
                parent2 = self.random.stdlib.choice(selected)
                child: Individual | None = None
                if self.random.stdlib.random() < self.cr_rate:
                    child = parent1.crossover(
                        self.random,
                        inst,
                        content_placement,
                        parent2,
                        self.cr_num_retries_per_req,
                    )
                    cr_success_rate.update(child is not None)
                if child is None:
                    child = parent1

                if self.random.stdlib.random() < self.mut_rate:
                    child = child.mutate(
                        self.random,
                        inst,
                        pathgen,
                        content_placement,
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
                    count=fitness_evaluator.fitness_eval_count,
                )
            )
            log.debug(
                SL(
                    "Best individual",
                    gen=gen,
                    indv=[
                        [r.node_list for r in routes]
                        for routes in self.best(
                            population, fitness_evaluator
                        ).all_routes
                    ],
                    fitness=self.best(population, fitness_evaluator).fitness(
                        fitness_evaluator
                    ),
                )
            )
            population = next_population
        return population, fitness_evaluator

    def best(
        self, collection: list[Individual], fitness_evaluator: FitnessEvaluator
    ) -> Individual:
        return min(collection, key=lambda ind: ind.fitness(fitness_evaluator))

    @override
    def route_instance(
        self, inst: Instance, content_placement: dict[int, set[int]]
    ) -> ilist[ilist[Route]]:
        population, fitness_evaluator = self.evolve(
            inst, content_placement, self.num_gens
        )
        return self.best(population, fitness_evaluator).all_routes

    @override
    def route_request(self, req: Request, top: Topology, dst: set[int]) -> ilist[Route]:
        raise NotImplementedError()
