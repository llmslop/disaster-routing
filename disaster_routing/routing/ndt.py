from typing import override
from .routing_algo import Route
import random


# intermediate individual representation for crossover and mutation
# https://www.sciencedirect.com/science/article/abs/pii/S0952197623016470
class NodeDepthTree:
    depths: list[int]

    def __init__(self, depths: list[int]):
        self.depths = depths

    @staticmethod
    def from_routes(routes: list[Route]) -> "NodeDepthTree":
        top = routes[0].top
        node_to_dz = {node: i for i, dz in enumerate(top.dzs) for node in dz.nodes}
        dz_depths: list[list[int]] = [[] for _ in range(len(top.dzs))]
        for route in routes:
            for depth, node in enumerate(route.node_list):
                dz_depths[node_to_dz[node]].append(depth)

        # TODO: handle the case where a DZ have more than one depths
        return NodeDepthTree([min(depths, default=-1) for depths in dz_depths])

    def copy(self) -> "NodeDepthTree":
        return NodeDepthTree(self.depths.copy())

    @override
    def __repr__(self) -> str:
        return str(self.depths)

    def crossover(self, other: "NodeDepthTree") -> "NodeDepthTree":
        # Single-point crossover
        point = random.randint(1, len(self.depths) - 1)
        child_depths = self.depths[:point] + other.depths[point:]
        child = NodeDepthTree.__new__(NodeDepthTree)
        child.depths = child_depths
        return child

    def mutate(self, mutation_rate: float) -> "NodeDepthTree":
        copy = self.copy()
        for i in range(len(copy.depths)):
            if random.random() < mutation_rate:
                # Mutate depth randomly within a reasonable range
                if copy.depths[i] != -1:
                    # For demonstration, add or subtract 1, but keep >= 0
                    change = random.choice([-1, 1])
                    new_depth = max(0, copy.depths[i] + change)
                    copy.depths[i] = new_depth
        return copy
