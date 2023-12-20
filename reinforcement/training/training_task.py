import torch.optim as optim
from .mse_loss import compute_swarm_loss

from .play_ground import PlayGround
from .training_report import TrainingReport
from .experience_buffer import ExperienceBuffer

from tqdm import trange
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # Common
    seed: int = 42
    batch_size: int = 32
    epoch_count: int = 40000
    steps_per_epoch: int = 1
    learning_rate: float = 0.0001

    # Evaluation
    evaluation_frequency: int = 100
    games_per_evaluation: int = 3
    steps_per_evaluation: int = 20

    loss_frequency: int = 20
    update_target_frequency: int = 100
    experience_buffer_size: int = 10000

    init_explore_rate: float = 1.0
    final_explore_rate: float = 0.1
    decay_steps: int = 10000


class TrainingTask:
    def __init__(self, pg: PlayGround, config: TrainingConfig):
        self.pg = pg
        self.config = config

        self.report = TrainingReport(pg.swarms)
        self.optimizer = optim.Adam(pg.parameters, lr=config.learning_rate)

    def run(self, buffer: ExperienceBuffer = None):

        if not buffer:
            buffer = ExperienceBuffer(size=self.config.experience_buffer_size)
            buffer.warm_up(self.pg)

        elif not buffer.is_valid(self.pg):
            raise RuntimeError("Buffer and current playground do not match")

        epoch = 0
        state = self.pg.new_game()

        with trange(epoch, self.config.epoch_count) as progress_bar:
            for epoch in progress_bar:

                loss_value, state = self.train_one_epoch(state, epoch, buffer)

                if epoch % self.config.loss_frequency == 0:
                    pass
                    # self.loss_report.append(loss_value)

                if epoch % self.config.update_target_frequency == 0:
                    self.pg.update_target_policy()

                if epoch % self.config.evaluation_frequency == 0:
                    reward = self.pg.evaluate(
                        n_games=self.config.games_per_evaluation,
                        n_steps=self.config.steps_per_evaluation)
                    self.report.append(reward)

    def train_one_epoch(self, state, epoch: int, buffer: ExperienceBuffer):

        e1, e2 = self.config.init_explore_rate, self.config.final_explore_rate
        self.pg.reduce_exploration(e1, e2, self.config.decay_steps, epoch)

        if self.pg.is_over:
            state = self.pg.new_game()

        reward, state = self.pg.play_game(state, buffer, self.config.steps_per_epoch)

        experience_batch = buffer.sample(self.config.batch_size)

        loss_values = []
        for swarm in self.pg.swarms:
            loss = compute_swarm_loss(swarm, experience_batch)
            loss.backward()
            loss_values.append(loss.data.cpu().item())

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss_values, state

    def get_report(self) -> TrainingReport:
        return self.report
