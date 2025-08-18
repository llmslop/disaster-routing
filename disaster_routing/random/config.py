from dataclasses import dataclass
from hydra.core.config_store import ConfigStore


@dataclass
class RandomConfig:
    pass


@dataclass
class SeededRandomConfig(RandomConfig):
    _target_: str = "disaster_routing.random.seeded.SeededRandom"
    seed: int = 42


@dataclass
class UnseededRandomConfig(RandomConfig):
    _target_: str = "disaster_routing.random.unseeded.UnseededRandom"


def register_random_configs(group: str = "random"):
    cs = ConfigStore.instance()
    cs.store(group=group, name="seeded", node=SeededRandomConfig)
    cs.store(group=group, name="unseeded", node=UnseededRandomConfig)
