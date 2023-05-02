import random
from typing import Any, Dict, List, Optional, Set

import torch
from datasets import Dataset
from torch import Tensor
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase

from ...utils.data import DataCollator


class CrossEncoderSentenceRetrievalDataCollator(DataCollator):
    @property
    def wiki_dataset(self):
        return self._wiki_dataset
    
    @wiki_dataset.setter
    def wiki_dataset(self, v: Dataset):
        self._wiki_dataset = v

        self._full_document_ids = set()
        self._full_sentence_ids = set()
        self._sentence_id_groups: Dict[int, Set[int]] = {}
        for i, x in enumerate(tqdm(self._wiki_dataset, desc='Creating Wiki Index')):
            self._full_document_ids.add(i)
            self._sentence_id_groups[i] = set()
            for j, _ in enumerate(x['lines']):
                self._full_sentence_ids.add((i, j))
                self._sentence_id_groups[i].add((i, j))

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__(tokenizer)

        assert self.tokenizer.sep_token
        self._wiki_dataset: Optional[Dataset] = None

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        assert self.wiki_dataset

        text = []
        score = []
        for x in batch:
            if x['sentence'] is None:
                x['excluded_document'] = set(x['excluded_document']) if x['excluded_document'] else set()
                x['excluded_sentence'] = set(tuple(si) for si in x['excluded_sentence']) if x['excluded_sentence'] else set()
                candidate_document_ids = self._full_document_ids - x['excluded_document']
                candidate_document_ids = list(candidate_document_ids)
                candidate_document_ids = random.choices(candidate_document_ids, k=min(10000, len(candidate_document_ids)))
                candidate_sentence_ids = set()
                for di in candidate_document_ids:
                    candidate_sentence_ids |= self._sentence_id_groups[di]
                candidate_sentence_ids -= x['excluded_sentence']
                candidate_sentence_ids = list(candidate_sentence_ids)
                x['sentence'] = random.choice(candidate_sentence_ids)

            di, li = x['sentence']
            sentence = self.wiki_dataset[di]['lines'][li]

            # if x['score_level'] == 0:
            #     s = random.uniform(0.0, 0.1)
            # elif x['score_level'] == 1:
            #     s = random.uniform(0.2, 0.4)
            # elif x['score_level'] == 2:
            #     s = random.uniform(0.8, 1.0)

            if x['score_level'] == 0:
                s = 0.0
            elif x['score_level'] == 1:
                s = 0.3
            elif x['score_level'] == 2:
                s = 1.0
            
            t = f'{x["claim"]} {self.tokenizer.sep_token} {sentence}'
            text.append(t)
            score.append(s)

        output = self.tokenizer(
            text,
            padding=True,
            max_length=512,
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=False
        )
        output['score'] = torch.tensor(score)
        return output
