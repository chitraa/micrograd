from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import nn

class TinyNN(nn.Module):
    def __init__(self):
        """ Layers(network) definition
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(3, 4),
            nn.Tanh(),
            nn.Linear(4, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)
    