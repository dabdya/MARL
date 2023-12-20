import numpy as np
import matplotlib.pyplot as plt
from typing import List

from ..communication.core import Swarm
from .utils import smooth


class TrainingReport:
    def __init__(self, swarms: List[Swarm]):
        self.swarms = swarms
        self._data = []

    @property
    def data(self):
        records = np.stack(self._data)
        n_records, n_agents = records.shape
        return records.reshape(n_agents, n_records)

    def append(self, record: np.array) -> None:
        """
        Parameters:
            record (np.array):
                A one-dimensional vector that stores the measurement for each agent
        """
        self._data.append(record)

    @staticmethod
    def smoothing(values: np.array) -> np.array:
        smoothed_values = []
        for i in range(len(values)):
            smoothed_values.append(smooth(values[i,]))

        return np.vstack(smoothed_values)

    def plot_for_each_agent(self, title: str = None, smoothing: bool = True) -> None:
        data = self.data
        if smoothing:
            data = TrainingReport.smoothing(data)

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.set_title(title)
        ax.set_xlabel('Epoch count scaled by measurement frequency')
        ax.set_ylabel(f'Mean data value')

        for swarm in self.swarms:
            for agent in swarm.squad:
                label = f'{agent.index}_{agent.__class__.__name__}'
                ax.plot(data[agent.index,], label=label)

        ax.grid()
        ax.legend()

    def plot_common_report(self, title: str = None, smoothing: bool = True) -> None:
        common_data = self.common_data()

        if smoothing:
            common_data = smooth(common_data)

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.set_title(title)
        ax.set_xlabel('Epoch count scaled by evaluation frequency')
        ax.set_ylabel('Mean value')

        ax.plot(common_data)
        ax.grid()

    def plot_swarm_report(self, title: str = None, smoothing: bool = True) -> None:
        swarm_data = self.swarm_data()

        if smoothing:
            swarm_data = TrainingReport.smoothing(swarm_data)

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.set_title(title)
        ax.set_xlabel('Epoch count scaled by evaluation frequency')
        ax.set_ylabel('Mean value')

        for i, swarm in enumerate(self.swarms):
            label = f'{i}_{swarm.__class__.__name__}'
            ax.plot(swarm_data[i,], label=label)

        ax.grid()
        ax.legend()

    def common_data(self) -> np.array:
        return self.data.mean(axis=0)

    def swarm_data(self) -> np.array:
        return np.vstack([
            self.data[swarm.indexes].mean(axis=0)
            for swarm in self.swarms
        ])
