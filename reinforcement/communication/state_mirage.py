import numpy as np
from .core import Swarm


class MirageState(Swarm):
    def __init__(self, *args, **kwargs):
        super(MirageState, self).__init__(*args, **kwargs)

    def get_action(self, swarm_state: np.array) -> np.array:
        pass
