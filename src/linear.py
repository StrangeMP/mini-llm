import torch
from torch import nn
import math


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,  # final dimension of the input
        out_features: int,  # final dimension of the output
        device: torch.device | None = None,  # Device to store the parameters on
        dtype: torch.dtype | None = None,  # Data type of the parameters
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), dtype=dtype, device=device)
        )
        sigma = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(
            self.weight, mean=0.0, std=sigma, a=-3.0 * sigma, b=3.0 * sigma
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return torch.einsum('...i,oi->...o', x, self.weight)
        return torch.matmul(x, self.weight.transpose(-1, -2))