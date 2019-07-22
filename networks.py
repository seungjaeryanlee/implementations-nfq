"""Networks for NFQ."""
import torch
import torch.nn as nn


class NFQNetwork(nn.Module):
    def __init__(self):
        """Networks for NFQ."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 5),
            nn.Sigmoid(),
            nn.Linear(5, 5),
            nn.Sigmoid(),
            nn.Linear(5, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of observation and action concatenated.

        Returns
        -------
        y : torch.Tensor
            Forward-propagated observation predicting Q-value.

        """
        return self.layers(x)
