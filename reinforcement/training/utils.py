from pathlib import Path
import yaml


def load_config(config_path: Path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config
