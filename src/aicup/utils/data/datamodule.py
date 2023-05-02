from abc import ABC
from typing import Optional

import lightning as L
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from .datacollator import DataCollator


class LightningDataModuleX(L.LightningDataModule, ABC):
    data_collator: Optional[DataCollator] = None

    @property
    def train_dataset(self):
        return self.dataset['train']
    
    @property
    def val_dataset(self):
        return self.dataset['test']

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        batch_size: int = 1,
        batch_size_val: Optional[int] = None,
        num_workers: int = 1,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()

        self.dataset_path = dataset_path
        self.tokenizer = tokenizer

        self.batch_size = batch_size
        self.batch_size_val = batch_size_val or batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset_path is not None:
            self.dataset = load_from_disk(self.dataset_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_val,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
        )
