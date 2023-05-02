import warnings
from typing import List

import safetensors
from torch import Tensor

__all__ = ['SafeOpen', 'TensorSlice']


class TensorSlice:
    def __init__(self, slice) -> None:
        self.slice = slice

    def get_shape(self) -> List[int]:
        return self.slice.get_shape()
    
    def __getitem__(self, indexing) -> Tensor:
        return self.slice[indexing]


class SafeOpen:
    def __init__(
        self,
        path: str,
        framework: str = 'pt',
        device: str = 'cpu',
    ) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            self.f = safetensors.safe_open(path, framework, device)

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.f.__exit__(exc_type, exc_value, traceback)
    
    def close(self):
        self.__exit__(None, None, None)

    def keys(self) -> List[str]:
        return self.f.keys()

    def get_tensor(self, key: str) -> Tensor:
        return self.f.get_tensor(key)
    
    def get_slice(self, key: str):
        return TensorSlice(self.f.get_slice(key))
