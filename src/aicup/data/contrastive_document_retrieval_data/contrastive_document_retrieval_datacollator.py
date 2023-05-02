import random
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from ...utils.data import DataCollator


class ContrastiveDocumentRetrieverDataCollator(DataCollator):
    @property
    def wiki_dataset(self):
        return self._wiki_dataset
    
    @wiki_dataset.setter
    def wiki_dataset(self, v: Dataset):
        self._wiki_dataset = v
        self._full_document_ids = set(range(len(v)))

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__(tokenizer)

        self._wiki_dataset: Optional[Dataset] = None

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        assert self.wiki_dataset
        output = {
            'claim': [],
            'document': [],
            'label': [],
        }

        for x in batch:
            if x['document'] is None:
                candidates = self._full_document_ids - set(x['excluded_documents'])
                x['document'] = random.choice(list(candidates))

            output['claim'].append(x['claim'])
            output['document'].append(self.wiki_dataset[x['document']]['text'])
            output['label'].append(x['label'])

        output['claim'] = self.tokenizer(output['claim'], padding=True, return_tensors='pt', return_token_type_ids=False)
        output['document'] = self.tokenizer(output['document'], padding=True, truncation=True, max_length=512, return_tensors='pt', return_token_type_ids=False)
        output['label'] = torch.tensor(output['label'])
        return output
