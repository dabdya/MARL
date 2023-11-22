import gymnasium as gym
from .predator_prey import PredatorPreyEnv


gym.register(
    id = "PredatorPrey",
    entry_point = "environments.predator_prey:PredatorPreyEnv"
)
