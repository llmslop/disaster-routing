from typing import override


class ModulationFormat:
    name: str
    rate: float
    reach: float
    power: float | None

    def __init__(self, name: str, rate: float, reach: float, power: float | None):
        self.name = name
        self.rate = rate
        self.reach = reach
        self.power = power

    def relative_bpsk_rate(self) -> float:
        return self.rate / ModulationFormat.bpsk().rate

    @classmethod
    def bpsk(cls) -> "ModulationFormat":
        return cls("BPSK", 12.5, 9600, 175.498)

    @classmethod
    def qpsk(cls) -> "ModulationFormat":
        return cls("QPSK", 25, 4800, 154.457)

    @classmethod
    def c8_qam(cls) -> "ModulationFormat":
        return cls("8-QAM", 37.5, 2400, 133.416)

    @classmethod
    def c16_qam(cls) -> "ModulationFormat":
        return cls("16-QAM", 50, 1200, 112.374)

    @classmethod
    def all(cls) -> list["ModulationFormat"]:
        return [cls.bpsk(), cls.qpsk(), cls.c8_qam(), cls.c16_qam()]

    @classmethod
    def best_rate_format_with_distance(
        cls, dist: float, formats: list["ModulationFormat"] | None = None
    ) -> "ModulationFormat | None":
        formats = formats if formats is not None else cls.all()
        return max(
            (format for format in formats if format.reach >= dist),
            key=lambda format: format.rate,
            default=None,
        )

    @override
    def __repr__(self) -> str:
        return self.name
