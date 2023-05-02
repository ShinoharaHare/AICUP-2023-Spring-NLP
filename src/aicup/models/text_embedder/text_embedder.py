from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn
from transformers import BatchEncoding


class TextEmbedder(nn.Module, ABC):

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        pass


    @abstractmethod
    def forward(self, encoding: BatchEncoding) -> Tensor:
        pass
