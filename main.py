from reinforcement.training import Agent, TrainingTask, PlayGround
from reinforcement.policy import PolicyFactory, SimpleFullyConnected
from reinforcement.communication import NoCommunication, SharePolicy, IC3Net

from environments import PredatorPreyEnv
from environments.utils import initizalize_env
from reinforcement.training.utils import load_config

from argparse import ArgumentParser
from pathlib import Path
import numpy as np


import warnings
warnings.filterwarnings("ignore")


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config", required = False, type = Path, 
        default = Path("./reinforcement/training/settings.yaml"))
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config = load_config(args.config)

    env = PredatorPreyEnv()
    initizalize_env(env, n_predators = 3, mode = "cooperative")

    state_dim = np.prod(env.observation_space.shape)
    n_actions = env.action_space.nvec.item()

    factory = PolicyFactory(state_dim, n_actions, hidden_size = 64)
    # factory = PolicyFactory(128, n_actions, hidden_size = 64)

    swarms = [
        SharePolicy([
            Agent(index, factory.get_policy(SimpleFullyConnected)) 
            for index in range(3)], best_neighbours = 3)
    ]
    pg = PlayGround(env, swarms)

    training_task = TrainingTask(pg, config)
    training_task.run()

    loss, reward = training_task.get_report()
    import pandas as pd

    pd.Series(loss).to_csv("loss.csv", header = None)
    pd.Series(reward).to_csv("reward.csv", header = None)
