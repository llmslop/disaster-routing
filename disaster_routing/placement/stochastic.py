from typing import override

from disaster_routing.instances.instance import Instance
from disaster_routing.utils.ilist import ilist
from disaster_routing.random.random import Random
from disaster_routing.placement.naive import NaiveContentPlacement


class StochasticContentPlacement(NaiveContentPlacement):
    random: Random

    def __init__(self, random: Random):
        self.random = random

    def shuffle_ilist(self, il: ilist[int]) -> ilist[int]:
        lst = list(il)
        self.random.stdlib.shuffle(lst)
        return tuple(lst)

    @override
    def sort_avail_dc_positions(
        self, avail_dcs: ilist[int], inst: Instance
    ) -> ilist[int]:
        return self.shuffle_ilist(avail_dcs)

    @override
    def sort_content_dc_positions(
        self, dcs: ilist[int], inst: Instance, content: int
    ) -> ilist[int]:
        return self.shuffle_ilist(dcs)
