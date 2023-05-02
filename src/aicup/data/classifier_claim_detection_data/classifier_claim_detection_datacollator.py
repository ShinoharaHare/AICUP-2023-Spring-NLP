import random
from typing import Any, Dict, List, Optional, Set

import torch
from datasets import Dataset
from torch import Tensor
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase

from ...utils.data import DataCollator


class ClassifierClaimDetectionDataCollator(DataCollator):
    @property
    def wiki_dataset(self):
        return self._wiki_dataset
    
    @wiki_dataset.setter
    def wiki_dataset(self, v: Dataset):
        self._wiki_dataset = v
        
        self._full_sentence_ids = []
        for i, x in enumerate(tqdm(self._wiki_dataset, desc='Creating Wiki Index')):
            for j, _ in enumerate(x['lines']):
                self._full_sentence_ids.append((i, j))

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__(tokenizer)

        assert self.tokenizer.sep_token
        self._wiki_dataset: Optional[Dataset] = None

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        assert self.wiki_dataset

        text = []
        label = []
        for x in batch:
            x['evidences'] = x['evidences'] or []
            if x['extra_sentences'] is None:
                x['excluded_sentences'] = set(tuple(si) for si in x['excluded_sentences']) if x['excluded_sentences'] else set()
                num_extra_sentences = random.randint(min(len(x['evidences']), 5), 5) - len(x['evidences'])
                candidates = random.choices(self._full_sentence_ids, k=10000)
                candidates = set(candidates)
                candidates -= x['excluded_sentences']
                candidates = list(candidates)
                x['extra_sentences'] = random.choices(candidates, k=num_extra_sentences)
            
            sentences = x['evidences'] + x['extra_sentences']
            # random.shuffle(sentences)
            sentences = [self.wiki_dataset[di]['lines'][li] for di, li in sentences]
            sentences = ' <sep> '.join(sentences)
            text.append(f'{x["claim"]} <sep> {sentences}')
            label.append(x['label'])
        
        output = self.tokenizer(
            text,
            padding=True,
            # padding='max_length',
            max_length=512,
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=False
        )
        output['label'] = torch.tensor(label)
        return output
