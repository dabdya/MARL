from ..communication.core import Swarm
from .experience_buffer import ExperienceBuffer

import numpy as np
import gymnasium as gym
from typing import List


class PlayGround:
    def __init__(self, environment: gym.Env, swarms: List[Swarm] = None):
        self._swarms = swarms
        self._environment = environment

    @property
    def agents(self):
        return [agent for swarn in self._swarms for agent in swarn.squad]

    def new_game(self):
        return self._environment.reset()

    @property
    def is_over(self):
        return self._environment.episode_over

    def play_game(self, start_state: np.array, buffer: ExperienceBuffer = None, n_steps: int = 1000):

        state = start_state
        total_reward = 0

        for _ in range(n_steps):

            action = self.get_action(state)
            next_state, reward, is_done, _ = self._environment.step(action)

            buffer.add(state, action, reward, next_state, is_done)

            total_reward += reward
            if is_done:
                state = self._environment.reset()
            else:
                state = next_state

        return total_reward, state

    def evaluate(self, n_games: int = 1, n_steps: int = 1000):

        rewards = []
        for _ in range(n_games):
            state = self._environment.reset()
            total_reward = 0
            for _ in range(n_steps):
                action = self.get_action(state)
                new_state, reward, is_done, *_ = self._environment.step(action)

                total_reward += reward
                if is_done:
                    break

                state = new_state

            rewards.append(total_reward)

        return np.mean(np.stack(rewards), axis=0)

    def get_action(self, environment_state: np.array):

        n_agents, *_ = environment_state.shape
        actions = np.zeros(n_agents)

        for swarm in self._swarms:
            agent_indexes = [agent.index for agent in swarm.squad]
            swarm_state = environment_state[agent_indexes,]
            actions[agent_indexes,] = swarm.get_action(swarm_state)

        return actions

    def update_target_policy(self):
        for agent in self.agents:
            agent.update_target_policy()

    def reduce_exploration(self, init_value, final_value, decay_steps, epoch: int):
        for agent in self.agents:
            agent.explore_rate = linear_decay(init_value, final_value, decay_steps, epoch)

    @property
    def parameters(self):
        params = []
        for agent in self.agents:
            params += agent.policy.parameters()
        return params


def linear_decay(init_val, final_val, cur_step, total_steps):
    if cur_step >= total_steps:
        return final_val
    return (init_val * (total_steps - cur_step) +
            final_val * cur_step) / total_steps
