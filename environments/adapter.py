from .predator_prey import PredatorPreyEnv
from argparse import ArgumentParser

from enum import Enum


class EnvironmentMode(Enum):
    Cooperative = 0
    Competitive = 1
    Mixed = 2


def init_environment(
        env: PredatorPreyEnv,
        dim: int,
        n_predators: int,
        mode: EnvironmentMode = EnvironmentMode.Cooperative
):
    env.init_curses()

    parser = ArgumentParser()
    env.init_args(parser)
    args = parser.parse_args([])
    args.nfriendly = n_predators
    args.mode = mode.name.lower()
    args.dim = dim
    env.multi_agent_init(args)
