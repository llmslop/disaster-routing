from disaster_routing.routing.routing_algo import Route, RoutingAlgorithm
from disaster_routing.utils.ilist import ilist


class CDPSolution:
    all_routes: ilist[ilist[Route]]
    start_indices: dict[int, int]
    num_fses: dict[int, int]
    content_placement_override: bool

    def __init__(
        self,
        all_routes: ilist[ilist[Route]],
        start_indices: dict[int, int],
        num_fses: dict[int, int],
        content_placement_override: bool = False,
    ):
        self.all_routes = RoutingAlgorithm.sort_routes(
            all_routes, num_fses, start_indices
        )
        self.start_indices = start_indices
        self.num_fses = num_fses
        self.content_placement_override = content_placement_override

    def total_fs(self) -> int:
        flattened_routes = [route for routes in self.all_routes for route in routes]
        return sum(
            len(route.edges()) * self.num_fses[i]
            for i, route in enumerate(flattened_routes)
        )

    def mofi(self) -> int:
        num_routes = sum(len(routes) for routes in self.all_routes)
        return max(self.start_indices[i] + self.num_fses[i] for i in range(num_routes))
