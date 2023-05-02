from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn
from transformers import BatchEncoding


class CrossEncoder(nn.Module, ABC):

    @abstractmethod
    def forward(self, encoding: BatchEncoding) -> Tensor:
        pass
