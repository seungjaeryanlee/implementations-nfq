"""Saving and loading models."""
import os


def tf_load(net, load_dir):
    """Load TensorFlow network.

    NOTE(seungjaeryanlee): Does not load optimizer.
    """
    import tensorflow as tf

    net = tf.keras.models.load_model(f"{ load_dir }/tf")

    return net


def tf_save(net, save_dir):
    """Save TensorFlow network.

    NOTE(seungjaeryanlee): Does not save optimizer.
    """
    # Create specified directory if it does not exist yet
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Includes optimizer
    # tf.keras.models.save_model(net, f"{ save_dir }/net", save_format="tf")
    import tensorflow as tf

    net = tf.keras.models.save_model(f"{ save_dir }/tf", save_format="tf")

    return net


def pt_load(net, optimizer, load_path):
    """Load PyTorch network."""
    import torch

    state_dict = torch.load(load_path)
    net.load_state_dict(state_dict["net"])
    optimizer.load_state_dict(state_dict["optimizer"])

    return net, optimizer


def pt_save(net, optimizer, save_path):
    """Save PyTorch network."""
    # Create specified directory if it does not exist yet
    save_dir = "/".join(save_path.split("/")[:-1])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    import torch

    torch.save(
        {"net": net.state_dict(), "optimizer": optimizer.state_dict()}, save_path
    )

    return net, optimizer
