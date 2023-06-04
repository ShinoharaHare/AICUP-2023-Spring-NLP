from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import Self


class PartiallyFrozenLinear(nn.Module):
    def __init__(self, linear: nn.Linear, pivot: int) -> None:
        super().__init__()

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.pivot = pivot

        self.w1 = nn.Parameter(linear.weight[:pivot], requires_grad=False)
        self.w2 = nn.Parameter(linear.weight[pivot:], requires_grad=True)
        
        if linear.bias:
            self.b1 = nn.Parameter(linear.bias[:pivot], requires_grad=False)
            self.b2 = nn.Parameter(linear.bias[pivot:], requires_grad=True)
        else:
            self.register_parameter('b1', None)
            self.register_parameter('b2', None)

        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)
        self._register_state_dict_hook(self._state_dict_hook)

    def forward(self, x: Tensor):
        x1 = F.linear(x, self.w1, self.b1)
        x2 = F.linear(x, self.w2, self.b2)
        return torch.cat([x1, x2], dim=-1)

    @staticmethod
    def _state_dict_hook(module, state_dict: Dict[str, Tensor], prefix: str, local_metadata):
        w1 = state_dict.pop(f'{prefix}w1')
        w2 = state_dict.pop(f'{prefix}w2')
        state_dict[f'{prefix}weight'] = torch.cat([w1, w2])

        if 'b1' in state_dict or 'b2' in state_dict:
            b1 = state_dict.pop(f'{prefix}b1')
            b2 = state_dict.pop(f'{prefix}b2')
            state_dict[f'{prefix}bias'] = torch.cat([b1, b2])
    
    def _load_state_dict_pre_hook(self, state_dict: Dict[str, Tensor], prefix: str, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        weight = state_dict.pop(f'{prefix}weight')
        state_dict[f'{prefix}w1'] = weight[:self.pivot]
        state_dict[f'{prefix}w2'] = weight[self.pivot:]

        if 'bias' in state_dict:
            bias = state_dict.pop(f'{prefix}bias')
            state_dict[f'{prefix}b1'] = bias[:self.pivot]
            state_dict[f'{prefix}b2'] = bias[self.pivot:]
