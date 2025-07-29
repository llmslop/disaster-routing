from random import shuffle
from typing import override

from ..instances.instance import Instance
from ..utils.ilist import ilist
from .naive import NaiveContentPlacement


class StochasticContentPlacement(NaiveContentPlacement):
    def shuffle_ilist(self, il: ilist[int]) -> ilist[int]:
        lst = list(il)
        shuffle(lst)
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
