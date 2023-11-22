import torch.optim as optim
import numpy as np
from .mse_loss import compute_loss, compute_swarm_loss
from typing import Dict

from .play_ground import PlayGround
from .experience_buffer import ExperienceBuffer

from tqdm import trange


class TrainingTask:
    def __init__(self, pg: PlayGround, config: Dict):
        self.pg = pg
        
        for attribute_name, attribute_value in config.items():
            setattr(self, attribute_name, attribute_value)

        self.optimizer = optim.Adam(pg.parameters, lr = self.learning_rate)

        self.loss_history = []
        self.reward_history = []

    def run(self, buffer: ExperienceBuffer = None):

        if not buffer:
            buffer = ExperienceBuffer(size = self.buffer_size)
            buffer.warm_up(self.pg)

        elif not buffer.is_valid(self.pg):
            raise RuntimeError("Buffer and current playground do not match")

        epoch = 0
        state = self.pg.new_game()

        with trange(epoch, self.epoch_count) as progress_bar:
            for epoch in progress_bar:

                loss_value, state = self.train_one_epoch(state, epoch, buffer)

                if epoch % self.loss_freq == 0:
                    self.loss_history.append(loss_value)

                if epoch % self.update_target_freq == 0:
                    self.pg.update_target_policy()

                if epoch % self.evaluation_freq == 0:
                    reward = self.pg.evaluate(n_games = 3, n_steps = 20)
                    self.reward_history.append(reward)


    def train_one_epoch(self, state, epoch: int, buffer: ExperienceBuffer):

        e1, e2 = self.init_explore_rate, self.final_explore_rate
        self.pg.reduce_exploration(e1, e2, self.decay_steps, epoch)

        if self.pg.is_over:
            state = self.pg.new_game()
            
        reward, state = self.pg.play_game(state, buffer, self.steps_per_epoch)
        
        experience_batch = buffer.sample(self.batch_size)

        loss_values = []
        for swarm in self.pg._swarms:
            loss = compute_swarm_loss(swarm, experience_batch)
            loss.backward()
            loss_values.append(loss.data.cpu().item())

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss_values, state

    def get_report(self):
        return self.loss_history, self.reward_history
