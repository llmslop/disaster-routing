import logging
from typing import override

from ..utils.structlog import SL
from ..utils.ilist import ilist
from ..instances.instance import Instance
from .strategy import ContentPlacementStrategy

log = logging.getLogger(__name__)


class NaiveContentPlacement(ContentPlacementStrategy):
    @override
    def place_content(self, inst: Instance) -> dict[int, list[int]]:
        dcs = self.sort_avail_dc_positions(inst.possible_dc_positions, inst)[
            : inst.dc_count
        ]
        max_paths = self.max_num_paths(inst)
        log.debug(SL("Max number of paths per content", max_paths=max_paths))
        return {
            content: list(self.sort_content_dc_positions(dcs, inst, content)[:num])
            for content, num in max_paths.items()
        }

    def sort_avail_dc_positions(
        self, avail_dcs: ilist[int], _inst: Instance
    ) -> ilist[int]:
        return avail_dcs

    def sort_content_dc_positions(
        self, dcs: ilist[int], _inst: Instance, _content: int
    ) -> ilist[int]:
        return dcs
