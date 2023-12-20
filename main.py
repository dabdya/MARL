from reinforcement.training import TrainingTask, TrainingConfig, PlayGround
from reinforcement.policy import PolicyFactory, SimpleFullyConnected

from reinforcement.communication import IC3NetBased, ShareState, SharePolicy
from reinforcement.communication.core import Agent
from reinforcement.communication.aggregation import ValueBasedAggregation, RandomAggregation
from reinforcement.communication.utils import generate_communication_matrix

from environments import PredatorPreyEnv
from environments.adapter import init_environment, EnvironmentMode
import numpy as np

import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":

    config = TrainingConfig()

    env = PredatorPreyEnv()
    init_environment(env, dim=5, n_predators=5, mode=EnvironmentMode.Cooperative)

    state_dim = np.prod(env.observation_space.shape)
    n_actions = env.action_space.nvec.item()

    factory = PolicyFactory(state_dim, n_actions, hidden_size=64)
    # factory = PolicyFactory(128, n_actions, hidden_size=64)

    communication_matrix = generate_communication_matrix(5)

    # swarms = [
    #     ShareState(squad=[
    #         Agent(index, factory.get_policy(SimpleFullyConnected))
    #         for index in range(5)], communication_matrix=communication_matrix, best_neighbors=3)
    # ]

    swarms = [
        SharePolicy(squad=[
            Agent(index, factory.get_policy(SimpleFullyConnected))
            for index in range(5)],
            communication_matrix=communication_matrix, neighbors_depth=2, aggregation=ValueBasedAggregation())
    ]

    # swarms = [
    #     IC3NetBased([
    #         Agent(index, factory.get_policy(SimpleFullyConnected))
    #         for index in range(5)], state_dim, hidden_size=128)
    # ]
    pg = PlayGround(env, swarms)

    training_task = TrainingTask(pg, config)
    training_task.run()

    report = training_task.get_report()

