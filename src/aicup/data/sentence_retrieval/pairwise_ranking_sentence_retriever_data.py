from typing import Any, Dict, List

import opencc
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from ...utils.data import DataCollator, DataModule

__all__ = ['PairwiseRankingSentenceRetrievalDataCollator', 'PairwiseRankingRetrievalDataModule']

class PairwiseRankingSentenceRetrievalDataCollator(DataCollator):
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__(tokenizer)

        assert self.tokenizer.sep_token

        self.t2s = opencc.OpenCC('t2s')

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        batch = self.convert_ld_to_dl(batch)
        
        new_batch = {}

        batch_positive_claim = []
        batch_positive_sentence = []

        batch_negative_claim = []
        batch_negative_sentence = []
        for positive_claim, positive_sentence, negative_claim, negative_sentence in zip(batch['positive_claim'], batch['positive_sentence'], batch['negative_claim'], batch['negative_sentence']):
            ps = f'{positive_sentence[0]} <TITLE> {positive_sentence[1]}'
            ns = f'{negative_sentence[0]} <TITLE> {negative_sentence[1]}'

            batch_positive_claim.append(self.t2s.convert(positive_claim))
            batch_positive_sentence.append(self.t2s.convert(ps))

            batch_negative_claim.append(self.t2s.convert(negative_claim))
            batch_negative_sentence.append(self.t2s.convert(ns))

        new_batch['positive'] = self.tokenizer(
            batch_positive_claim,
            batch_positive_sentence,
            return_tensors='pt',
            padding=True,
            # padding='max_length',
            max_length=1024,
            truncation=True,
        )

        new_batch['negative'] = self.tokenizer(
            batch_negative_claim,
            batch_negative_sentence,
            return_tensors='pt',
            padding=True,
            # padding='max_length',
            max_length=1024,
            truncation=True,
        )

        new_batch['labels'] = torch.ones(new_batch['positive']['input_ids'].size(0), dtype=torch.long)
        return new_batch


class PairwiseRankingRetrievalDataModule(DataModule):
    data_collator: PairwiseRankingSentenceRetrievalDataCollator
    data_collator_class = PairwiseRankingSentenceRetrievalDataCollator
