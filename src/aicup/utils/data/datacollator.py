from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

from torch import Tensor
import torch
from transformers import PreTrainedTokenizerBase


class DataCollator(ABC):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase
    ) -> None:
        self.tokenizer = tokenizer

    @abstractmethod
    def __call__(self, batch: List[Any]) -> Dict[str, Tensor]:
        pass
    
    @staticmethod
    def convert_ld_to_dl(ld: Dict[str, List]) -> Dict[str, List[Any]]:
        return {k: [d[k] for d in ld] for k in ld[0]}

    # @staticmethod
    # def convert_to_tensor_with_padding(l: List[List[Union[int, float]]], padding_value: Union[int, float]) -> Tensor:
    #     n = max(len(x) for x in l)
    #     l = [x + [padding_value] * (len(x) - n) for x in l]
    #     print(l)
    #     return torch.tensor(l)
