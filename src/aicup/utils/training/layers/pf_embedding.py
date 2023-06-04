import torch
from torch import LongTensor, nn


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

        self.embedding1 = nn.Embedding.from_pretrained(
            embeddings.weight.data[:self.pivot],
            freeze=True,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )
        self.embedding2 = nn.Embedding.from_pretrained(
            embeddings.weight.data[self.pivot:],
            freeze=False,
            padding_idx=None,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

    def forward(self, x: LongTensor):
        mask = x < self.pivot
        device = x.device
        e = torch.empty(*x.shape, self.embedding_dim, device=device)
        e[mask] = self.embedding1(x[mask]).to(e.dtype)
        e[~mask] = self.embedding2(x[~mask] - self.pivot).to(e.dtype)
        return e
