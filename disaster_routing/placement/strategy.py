from abc import ABC, abstractmethod

from ..instances.request import Request
from ..instances.instance import Instance


class ContentPlacementStrategy(ABC):
    @abstractmethod
    def place_content(self, inst: Instance) -> dict[int, set[int]]: ...

    def verify_placement(self, inst: Instance, placement: dict[int, set[int]]):
        dcs: set[int] = set()
        for _, content_dcs in placement.items():
            dcs.update(content_dcs)
            assert len(content_dcs) >= 2
        assert len(dcs) <= inst.dc_count
        assert all(dc in inst.possible_dc_positions for dc in dcs)

    def max_num_paths(self, inst: Instance) -> dict[int, int]:
        contents = self.content_to_requests_dict(inst)
        return {
            content: min(
                inst.dc_count,
                min(int(inst.topology.graph.in_degree[req.source]) for req in reqs),
            )
            for content, reqs in contents.items()
        }

    def content_to_requests_dict(self, inst: Instance) -> dict[int, list[Request]]:
        return {
            content: [req for req in inst.requests if req.content_id == content]
            for content in set(req.content_id for req in inst.requests)
        }
