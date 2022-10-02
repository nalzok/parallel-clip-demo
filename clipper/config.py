from dataclasses import dataclass
import flax.linen as nn


class MLP(nn.Module):    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(120)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(84)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(10)(x)
        return x


@dataclass
class Config:
    model: nn.Module
    epochs: int
    batch_size: int
    lr: float


config = Config(MLP(), 16, 512, 0.001)
