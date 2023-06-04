import json
import os
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from datasets import Dataset, load_dataset, load_from_disk
from elasticsearch import Elasticsearch
from torch.utils.data import Dataset as TorchDataset

from aicup.utils import NounExtractor, disable_datasets_cache


class WikiDataset(TorchDataset):
    @property
    def noun_extractor(self):
        self._noun_extractor = self._noun_extractor or NounExtractor()
        return self._noun_extractor

    def __init__(
        self,
        dataset: Dataset,
        t2i: Optional[Mapping[str, int]] = None,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.t2i = t2i or {x['id']: i for i, x in enumerate(dataset)}

        self._noun_extractor = None

    def __len__(self):
        return self.dataset.num_rows
    
    def __getitem__(self, index: Union[int, str]):
        if isinstance(index, str):
            index = self.t2i[index]

        return self.dataset[index]
    
    def has(self, title: str):
        return title in self.t2i
    
    def add_elasticsearch_index(
        self,
        hosts: Optional[str] = None,
        ca_certs: Optional[str] = None,
        basic_auth: Optional[Tuple[str, str]] = None,
    ):
        es_client = Elasticsearch(
            hosts=hosts,
            ca_certs=ca_certs,
            basic_auth=basic_auth
        )
        self.dataset.add_elasticsearch_index('text', 'wiki-text', es_client=es_client)
    
    def load_elasticsearch_index(
        self,
        hosts: Optional[str] = None,
        ca_certs: Optional[str] = None,
        basic_auth: Optional[Tuple[str, str]] = None,
    ):
        if basic_auth is None:
            username = os.environ.get('ES_USERNAME', None)
            password = os.environ.get('ES_PASSWORD', None)
            if username and password:
                basic_auth = (username, password)

        es_client = Elasticsearch(
            hosts=hosts or os.environ.get('ES_HOSTS', None),
            ca_certs=ca_certs or os.environ.get('ES_CA_CERTS', None),
            basic_auth=basic_auth,
        )
        self.dataset.load_elasticsearch_index('text', 'wiki-text', es_client=es_client)

    def retrieve(
        self,
        queries: List[str],
        top_k: int = 3,
        min_score: float = 10.0,
        return_by_noun: bool = True,
        merge_adjacent: bool = True,
        return_unmerged: bool = True,
    ) -> List[Dict[str, Any]]:
        batch_scores, batch_examples = self.dataset.get_nearest_examples_batch('text', queries, k=top_k)
        for i, (scores, examples) in enumerate(zip(batch_scores, batch_examples)):
            batch_examples[i] = {k: [x for s, x in zip(scores, examples[k]) if s >= min_score] for k in examples}

        if return_by_noun:
            batch_nouns = self.noun_extractor.extract(queries, merge_adjacent, return_unmerged)
            for nouns, examples in zip(batch_nouns, batch_examples):
                titles = set(n for n in nouns if self.has(n) and n not in examples['id'])
                for t in titles:
                    for k, v in self[t].items():
                        examples[k].append(v)
        return batch_examples

    def save(self, path: str):
        self.dataset.save_to_disk(os.path.join(path, 'dataset'))
        with open(os.path.join(path, 't2i.json'), 'w', encoding='utf-8') as f:
            json.dump(self.t2i, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        wiki_dataset = load_from_disk(os.path.join(path, 'dataset'))
        with open(os.path.join(path, 't2i.json'), 'r', encoding='utf-8') as f:
            t2i = json.load(f)
        return cls(wiki_dataset, t2i)

    @classmethod
    def from_json(cls, path: str, num_proc: Optional[int] = None):
        with disable_datasets_cache():
            wiki_dataset = load_dataset('json', data_dir=path)['train']
            def mapper(x: Dict[str, str]):
                x['raw_lines'] = x['lines']
                x['lines'] = re.split(r'^\d+\t', x['raw_lines'], flags=re.MULTILINE)[1:]
                x['lines'] = [l.strip() for l in x['lines']]
                return x
            wiki_dataset = wiki_dataset.map(mapper, num_proc=num_proc)
        return cls(wiki_dataset)
