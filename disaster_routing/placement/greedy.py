import logging
from statistics import mean
from typing import override

import networkx as nx

from disaster_routing.instances.instance import Instance
from disaster_routing.placement.naive import NaiveContentPlacement
from disaster_routing.utils.ilist import ilist
from disaster_routing.utils.structlog import SL

log = logging.getLogger(__name__)


class GreedyContentPlacement(NaiveContentPlacement):
    def get_path_lengths(self, inst: Instance) -> dict[int, dict[int, int]]:
        return dict(nx.shortest_path_length(inst.topology.graph))

    def sort_dict(self, d: dict[int, float]) -> dict[int, float]:
        return dict(sorted(d.items(), key=lambda item: item[1]))

    @override
    def sort_avail_dc_positions(
        self, avail_dcs: ilist[int], inst: Instance
    ) -> ilist[int]:
        lengths = self.get_path_lengths(inst)
        dc_dists = self.sort_dict(
            {
                dc: mean((lengths[dc][req.source] for req in inst.requests))
                for dc in avail_dcs
            }
        )
        log.debug(
            SL("Average distances from DCs to request source nodes", dc_dists=dc_dists)
        )
        return tuple(sorted(avail_dcs, key=lambda dc: dc_dists[dc]))

    @override
    def sort_content_dc_positions(
        self, dcs: ilist[int], inst: Instance, content: int
    ) -> ilist[int]:
        lengths = self.get_path_lengths(inst)
        requests = [req for req in inst.requests if req.content_id == content]
        dc_dists = self.sort_dict(
            {dc: mean((lengths[dc][req.source] for req in requests)) for dc in dcs}
        )
        log.debug(
            SL(
                "Average distances from DCs to source nodes of a specific content",
                content=content,
                dc_dists=dc_dists,
            )
        )
        return tuple(sorted(dcs, key=lambda dc: dc_dists[dc]))
