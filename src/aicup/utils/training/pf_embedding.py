from typing import Dict

import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor, nn


class PartiallyFrozenEmbedding(nn.Module):
    def __init__(
        self,
        embeddings: nn.Embedding,
        pivot: int,
    ) -> None:
        super().__init__()

        self.num_embeddings = embeddings.num_embeddings
        self.embedding_dim = embeddings.embedding_dim
        self.padding_idx = embeddings.padding_idx
        self.max_norm = embeddings.max_norm
        self.norm_type = embeddings.norm_type
        self.scale_grad_by_freq = embeddings.scale_grad_by_freq
        self.sparse = embeddings.sparse

        self.pivot = pivot

        self.w1 = nn.Parameter(embeddings.weight.data[:pivot], requires_grad=False)
        self.w2 = nn.Parameter(embeddings.weight.data[pivot:], requires_grad=True)

        # self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)
        # self._register_state_dict_hook(self._state_dict_hook)

    def forward(self, x: LongTensor):
        mask = x < self.pivot
        e = torch.empty(*x.shape, self.embedding_dim, device=self.w1.device, dtype=self.w1.dtype)
        e[mask] = F.embedding(x[mask], self.w1, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        e[~mask] = F.embedding(x[~mask] - self.pivot, self.w2, None, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse).to(e.dtype)
        return e
    
    @staticmethod
    def _state_dict_hook(module: "PartiallyFrozenEmbedding", state_dict: Dict[str, Tensor], prefix: str, local_metadata):
        w1 = state_dict.pop(f'{prefix}w1')
        w2 = state_dict.pop(f'{prefix}w2')
        state_dict[f'{prefix}weight'] = torch.cat([w1, w2])
    
    def _load_state_dict_pre_hook(self, state_dict: Dict[str, Tensor], prefix: str, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        weight = state_dict.pop(f'{prefix}weight')
        state_dict[f'{prefix}w1'] = weight[:self.pivot]
        state_dict[f'{prefix}w2'] = weight[self.pivot:]
