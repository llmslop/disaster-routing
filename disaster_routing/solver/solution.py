from disaster_routing.routing.routing_algo import Route, RoutingAlgorithm
from disaster_routing.utils.ilist import ilist


class CDPSolution:
    all_routes: ilist[ilist[Route]]
    start_indices: dict[int, int]
    num_fses: dict[int, int]

    def __init__(
        self,
        all_routes: ilist[ilist[Route]],
        start_indices: dict[int, int],
        num_fses: dict[int, int],
    ):
        self.all_routes = RoutingAlgorithm.sort_routes(all_routes)
        self.start_indices = start_indices
        self.num_fses = num_fses

    def total_fs(self) -> int:
        flattened_routes = [r for route in self.all_routes for r in route]
        return sum(
            self.num_fses[i] * len(flattened_routes[i].edges())
            for i in range(len(flattened_routes))
        )

    def mofi(self) -> int:
        num_routes = sum(len(routes) for routes in self.all_routes)
        return max(self.start_indices[i] + self.num_fses[i] for i in range(num_routes))
