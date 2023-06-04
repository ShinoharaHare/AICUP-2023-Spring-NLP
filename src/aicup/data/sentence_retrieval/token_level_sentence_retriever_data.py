from typing import Any, Dict, List

import opencc
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from ...utils.data import DataCollator, DataModule

__all__ = ['TokenLevelSentenceRetrievalDataCollator', 'TokenLevelSentenceRetrievalDataModule']

def rindex(l: list, x):
    for i in range(len(l) - 1, -1, -1):
        if l[i] == x:
            return i
    raise ValueError()


class TokenLevelSentenceRetrievalDataCollator(DataCollator):
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__(tokenizer)

        assert self.tokenizer.sep_token

        self.t2s = opencc.OpenCC('t2s')

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        batch = self.convert_ld_to_dl(batch)
        
        new_batch = {}

        batch_claim = [self.t2s.convert(c) for c in batch['claim']]
        batch_document = []
        batch_sentence_slices = []

        for sentences in batch['sentences']:
            document = ''
            
            s_ti = 0
            slices = []
            for s in sentences:
                document += self.t2s.convert(s) + '\n'
                e_ti = s_ti + len(self.tokenizer.tokenize(s))
                slices.append([s_ti, e_ti])
                s_ti = e_ti

            batch_document.append(document)
            batch_sentence_slices.append(slices)

        batch_encoding = self.tokenizer(
            batch_claim,
            batch_document,
            return_tensors='pt',
            truncation=True,
            max_length=4096,
            padding=True,
            # padding='max_length',
        )
        new_batch.update(batch_encoding)
        new_batch['claim_mask'] = torch.zeros_like(batch_encoding['input_ids'], dtype=torch.bool)
        new_batch['labels'] = torch.zeros_like(batch_encoding['input_ids'], dtype=torch.float)

        for bi, (sentence_slices, is_evidence) in enumerate(zip(batch_sentence_slices, batch['is_evidence'])):
            new_batch['labels'][bi, 0] = any(is_evidence)
            
            sequence_ids = batch_encoding.sequence_ids(bi)
            s_ti = sequence_ids.index(0)
            e_ti = rindex(sequence_ids, 0)
            new_batch['claim_mask'][bi, s_ti:e_ti + 1] = True
            
            try:
                offset = sequence_ids.index(1)
            except ValueError:
                continue
            
            for i, ((s_ti, e_ti), b) in enumerate(zip(sentence_slices, is_evidence)):
                sentence_slices[i] = slice(s_ti + offset, e_ti + offset)
                new_batch['labels'][bi, sentence_slices[i]] = b

        new_batch['sentence_slices'] = batch_sentence_slices
        new_batch['is_evidence'] = batch['is_evidence']
        
        return new_batch


class TokenLevelSentenceRetrievalDataModule(DataModule):
    data_collator: TokenLevelSentenceRetrievalDataCollator
    data_collator_class = TokenLevelSentenceRetrievalDataCollator
