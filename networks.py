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

        # Initialize weights to [-0.5, 0.5]
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.uniform_(m.weight, -0.5, 0.5)
                # TODO(seungjaeryanlee): What about bias?

        self.layers.apply(init_weights)

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
