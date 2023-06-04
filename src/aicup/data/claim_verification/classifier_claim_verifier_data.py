import re
from typing import Any, Dict, List

import opencc
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from ...utils.data import DataCollator, DataModule

__all__ = ['ClassifierClaimVerificationDataCollator', 'ClassifierClaimVerificationDataModule']

class ClassifierClaimVerificationDataCollator(DataCollator):
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__(tokenizer)

        assert self.tokenizer.sep_token
        assert '<TITLE>' in self.tokenizer.get_vocab()

        self.t2s = opencc.OpenCC('t2s')

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        batch = self.convert_ld_to_dl(batch)
        
        new_batch = {}

        batch_claim = [self.t2s.convert(c) for c in batch['claim']]
        batch_evidence = []

        for titles, sentences in zip(batch['titles'], batch['sentences']):
            for i in range(len(titles)):
                sentence = re.sub(r'(\t.*)+$', '', sentences[i])
                sentences[i] = f'{titles[i]} <TITLE> {sentence}'
            
            evidence = f' {self.tokenizer.sep_token} '.join(sentences)
            evidence = self.t2s.convert(evidence)
            batch_evidence.append(evidence)

        batch_encoding = self.tokenizer(
            batch_claim,
            batch_evidence,
            return_tensors='pt',
            padding=True,
            # padding='max_length',
            truncation=self.tokenizer.model_max_length < 1e30,
        )
        new_batch.update(batch_encoding)
        new_batch['labels'] = torch.tensor(batch['label'])
        return new_batch

class ClassifierClaimVerificationDataModule(DataModule):
    data_collator: ClassifierClaimVerificationDataCollator
    data_collator_class = ClassifierClaimVerificationDataCollator
