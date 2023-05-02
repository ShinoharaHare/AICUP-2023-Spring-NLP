from datasets import load_from_disk
from transformers import PreTrainedTokenizerBase

from ...utils.data import LightningDataModuleX
from .cross_encoder_sentence_retrival_datacollator import \
    CrossEncoderSentenceRetrievalDataCollator


class CrossEncoderSentenceRetrievalRetrievalDataModule(LightningDataModuleX):
    data_collator: CrossEncoderSentenceRetrievalDataCollator
    
    def __init__(
        self,
        dataset_path: str,
        wiki_dataset_path: str,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 1,
        batch_size_val: int | None = None,
        num_workers: int = 1,
        pin_memory: bool = True
    ) -> None:
        super().__init__(dataset_path, tokenizer, batch_size, batch_size_val, num_workers, pin_memory)

        self.wiki_dataset_path = wiki_dataset_path

        self.data_collator = CrossEncoderSentenceRetrievalDataCollator(tokenizer)

    def setup(self, stage: str | None = None) -> None:
        super().setup(stage)

        self.data_collator.wiki_dataset = load_from_disk(self.wiki_dataset_path)
