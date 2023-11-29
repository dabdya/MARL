from pathlib import Path
import numpy as np
import yaml
from scipy.signal import fftconvolve, gaussian


def load_config(config_path: Path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config


def smooth(values):
    kernel = gaussian(100, std=100)
    kernel = kernel / np.sum(kernel)
    return fftconvolve(values, kernel, 'valid')
