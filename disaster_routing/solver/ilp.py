from disaster_routing.eval.evaluator import Evaluator
from disaster_routing.ilp.cdp import ILPCDP
from disaster_routing.instances.instance import Instance
from disaster_routing.solver.solution import CDPSolution
from disaster_routing.solver.solver import CDPSolver


class ILPCDPSolver(CDPSolver):
    msg: bool

    def __init__(self, evaluator: Evaluator, msg: bool):
        super().__init__(evaluator)
        self.msg = msg

    def name(self) -> str:
        return "ilp"

    def solve(
        self, inst: Instance, content_placement: dict[int, set[int]]
    ) -> CDPSolution:
        ilp = ILPCDP(inst, self.evaluator)
        ilp.solve()
        raise NotImplementedError
