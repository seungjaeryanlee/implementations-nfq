"""Saving and loading models."""
import os


def pt_load(net, optimizer, load_dir):
    """Load PyTorch network and optimizer."""
    import torch

    state_dict = torch.load(f"{ load_dir }/cartpole.pt")
    net.load_state_dict(state_dict["net"])
    optimizer.load_state_dict(state_dict["optimizer"])

    return net, optimizer


def tf_load(net, load_dir):
    """Load TensorFlow network.

    NOTE(seungjaeryanlee): Does not load optimizer.
    """
    import tensorflow as tf

    net = tf.keras.models.load_model(f"{ load_dir }/cartpole")

    return net


def pt_save(net, optimizer, save_dir):
    """Save PyTorch network and optimizer."""
    # Create specified directory if it does not exist yet
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    import torch

    torch.save(
        {"net": net.state_dict(), "optimizer": optimizer.state_dict()},
        f"{ save_dir }/cartpole.pt",
    )

    return net, optimizer


def tf_save(net, save_dir):
    """Save TensorFlow network.

    NOTE(seungjaeryanlee): Does not save optimizer.
    """
    # Create specified directory if it does not exist yet
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    import tensorflow as tf

    net = tf.keras.models.save_model(f"{ save_dir }/cartpole", save_format="tf")

    return net
