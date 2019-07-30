"""Save and load trained PyTorch models."""
import os

import torch


def load_models(load_path: str, **kwargs):
    """Load specified models.

    Parameters
    ----------
    load_path : str
        Load path including the filename.

    """
    state_dict = torch.load(load_path)
    for key, value in kwargs.items():
        value.load_state_dict(state_dict[key])


def save_models(save_path: str, **kwargs):
    """Save specified models.

    Parameters
    ----------
    save_path : str
        Save path including the filename.

    """
    # Create specified directory if it does not exist yet
    save_dir = "/".join(save_path.split("/")[:-1])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save({key: value.state_dict() for key, value in kwargs.items()}, save_path)
