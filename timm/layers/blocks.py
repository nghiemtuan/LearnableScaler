import torch
import torch.nn as nn


class SkipBlock(nn.Module):
    def __init__( self, a: float, b: float, c: float):
        super().__init__()
        self.a = nn.Parameter( torch.Tensor( [ a ] ) )
        self.b = nn.Parameter( torch.Tensor( [ b ] ) )
        self.c = nn.Parameter( torch.Tensor( [ c ] ) )

    def forward(self, x: torch.Tensor, block: torch.Tensor ) -> torch.Tensor:
        return x + block
        return torch.clamp(self.a, max = 1) * x + self.b * block + self.c
