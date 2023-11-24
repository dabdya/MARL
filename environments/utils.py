from .predator_prey import PredatorPreyEnv
from argparse import ArgumentParser


def initizalize_env(
        env: PredatorPreyEnv, n_predators: int, mode: str = "cooperative") -> None:
    env.init_curses()

    parser = ArgumentParser(description = "Environment parser")
    env.init_args(parser)
    env.multi_agent_init(parser.parse_args(args=[]))
    env.npredator = n_predators
    env.mode = mode
