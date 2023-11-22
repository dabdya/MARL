from reinforcement.training import TrainingTask, PlayGround
from reinforcement.communication import NoCommunication, SharePolicy, IC3Net
from reinforcement.policy import PolicyFactory, SimpleFullyConnected
from reinforcement.training import Agent

from argparse import ArgumentParser
import numpy as np
from pathlib import Path

from environments import PredatorPreyEnv
import yaml

import warnings
warnings.filterwarnings("ignore")


def get_args(): # добавить выбор среды из имеющихся
    parser = ArgumentParser()
    parser.add_argument("--config", required = False, 
                        type = Path, default = Path("./reinforcement/training/settings.yaml"))
    return parser.parse_args()


def load_config(config_path: Path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config


def initizalize_env(env: PredatorPreyEnv, n_predators: int) -> None:
    env.init_curses()

    parser = ArgumentParser(description = "Environment parser")
    env.init_args(parser)
    env.multi_agent_init(parser.parse_args(args=[]))
    env.npredator = n_predators
    env.mode = "cooperative"


if __name__ == "__main__":
    args = get_args()
    config = load_config(args.config)

    env = PredatorPreyEnv()
    initizalize_env(env, n_predators = 3)

    state_dim = np.prod(np.array(env.observation_space.shape ))
    n_actions = env.action_space.nvec[0]

    factory = PolicyFactory(state_dim, n_actions, hidden_size = 64)
    factory2 = PolicyFactory(128, n_actions, hidden_size = 64)

    swarms = [
        IC3Net([
            Agent(0, factory2.get_policy(SimpleFullyConnected)),
            Agent(1, factory2.get_policy(SimpleFullyConnected)),
            Agent(2, factory2.get_policy(SimpleFullyConnected)),
        ], state_dim, hidden_size = 128
        ),
        # NoCommunication([
        #     Agent(0, factory.get_policy(SimpleFullyConnected)),
        #     Agent(1, factory.get_policy(SimpleFullyConnected)),
        # ]),
        # SharePolicy([
        #     Agent(0, factory.get_policy(SimpleFullyConnected)),
        #     Agent(1, factory.get_policy(SimpleFullyConnected)),
        # ])
    ]
    pg = PlayGround(env, swarms)

    training_task = TrainingTask(pg, config)
    training_task.run()

    loss, reward = training_task.get_report()
    import pandas as pd

    pd.Series(loss).to_csv("loss.csv", header = None)
    pd.Series(reward).to_csv("reward.csv", header = None)
