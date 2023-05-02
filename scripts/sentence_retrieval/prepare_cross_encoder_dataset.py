import json
import random
from typing import Any, Dict, List

import fire
from datasets import disable_caching, load_from_disk
from tqdm.auto import tqdm
from contextlib import contextmanager

@contextmanager
def efficient_difference(s1: set, s2: set):
    removed = s1 & s2
    s1 -= s2
    yield s1
    s1 |= removed


def main(
    dataset_path: str,
    wiki_dataset_path: str,
    output_dir: str,
):
    disable_caching()
    random.seed(42)

    dataset = load_from_disk(dataset_path)
    wiki_dataset = load_from_disk(wiki_dataset_path)

    document_mapping = {}
    sentence_mapping = {}
    
    sentence_id_groups: Dict[int, set] = {}
    full_document_ids = set()
    full_sentence_ids = set()
    for i, x in enumerate(tqdm(wiki_dataset, desc='Generating Wiki Mapping', leave=False)):
        document_mapping[x['id']] = i
        sentence_id_groups[i] = set()
        full_document_ids.add(i)
        for j, _ in enumerate(x['lines']):
            sentence_mapping[(x['id'], j)] = (i, j)
            full_sentence_ids.add((i, j))
            sentence_id_groups[i].add((i, j))
    
    def mapper(batch: Dict[str, List[Any]], is_test: bool = False):
        datapoints = {
            'claim': [],
            'sentence': [],
            'score_level': [],
            'excluded_document': [],
            'excluded_sentence': []
        }

        for i in range(len(batch['id'])):
            document_ids = set()
            sentence_ids = set()

            if batch['label'][i] != 2:
                for eg in batch['evidence'][i]:
                    for d, l in zip(eg['document'], eg['line']):
                        di = document_mapping[d]
                        document_ids.add(document_mapping[d])
                        sentence_ids.add(sentence_mapping[(d, l)])
                
            for si in sentence_ids:
                datapoints['claim'].append(batch['claim'][i])
                datapoints['sentence'].append(si)
                datapoints['score_level'].append(2)
                datapoints['excluded_document'].append(None)
                datapoints['excluded_sentence'].append(None)

            level_1_candidates = set()
            for di in document_ids:
                level_1_candidates |= sentence_id_groups[di]
            level_1_candidates -= sentence_ids

            num_level_0 = max(1, len(sentence_ids))
            num_level_1 = min(num_level_0, len(level_1_candidates))
            if is_test:
                with efficient_difference(full_document_ids, document_ids) as document_candidates:
                    document_candidates = random.choices(list(document_candidates), k=10000)

                level_0_candidates = set()
                for di in document_candidates:
                    level_0_candidates |= sentence_id_groups[di]
                level_0_candidates = list(level_0_candidates)

                level_1_candidates = list(level_1_candidates)

                for si in random.choices(level_0_candidates, k=num_level_0):
                    datapoints['claim'].append(batch['claim'][i])
                    datapoints['sentence'].append(si)
                    datapoints['score_level'].append(0)
                    datapoints['excluded_document'].append(None)
                    datapoints['excluded_sentence'].append(None)

                for si in random.choices(level_1_candidates, k=num_level_1):
                    datapoints['claim'].append(batch['claim'][i])
                    datapoints['sentence'].append(si)
                    datapoints['score_level'].append(1)
                    datapoints['excluded_document'].append(None)
                    datapoints['excluded_sentence'].append(None)
            else:
                for _ in range(num_level_0):
                    datapoints['claim'].append(batch['claim'][i])
                    datapoints['sentence'].append(None)
                    datapoints['score_level'].append(0)
                    datapoints['excluded_document'].append(document_ids)
                    datapoints['excluded_sentence'].append(None)

                for _ in range(num_level_1):
                    datapoints['claim'].append(batch['claim'][i])
                    datapoints['sentence'].append(None)
                    datapoints['score_level'].append(1)
                    datapoints['excluded_document'].append(None)
                    datapoints['excluded_sentence'].append(sentence_ids)
        return datapoints

    dataset['train'] = dataset['train'].map(mapper, remove_columns=dataset['train'].column_names, batched=True, batch_size=1)
    dataset['test'] = dataset['test'].map(mapper, fn_kwargs=dict(is_test=True), remove_columns=dataset['test'].column_names, batched=True, batch_size=1)

    dataset.save_to_disk(output_dir)


if __name__ == '__main__':
    fire.Fire(main)
