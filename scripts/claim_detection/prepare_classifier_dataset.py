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

    sentence_mapping = {}
    full_sentence_ids = set()
    for i, x in enumerate(tqdm(wiki_dataset, desc='Creating Wiki Mapping', leave=False)):
        for j, _ in enumerate(x['lines']):
            sentence_mapping[(x['id'], j)] = (i, j)
            full_sentence_ids.add((i, j))
    
    def mapper(batch: Dict[str, List[Any]], is_test: bool = False):
        datapoints = {
            'claim': [],
            'evidences': [],
            'extra_sentences': [],
            'label': [],
            'excluded_sentences': []
        }

        for i in range(len(batch['id'])):
            sentence_ids = set()
            evidence_group = batch['evidence'][i] if batch['label'][i] != 2 else []
            num_nei = max(1, len(evidence_group))

            for eg in evidence_group:
                for d, l in zip(eg['document'], eg['line']):
                    sentence_ids.add(sentence_mapping[(d, l)])

            if is_test:
                with efficient_difference(full_sentence_ids, sentence_ids) as candadates:
                    candadates = list(candadates)
                    for eg in evidence_group:
                        evidences = set(sentence_mapping[(d, l)] for d, l in zip(eg['document'], eg['line']))
                        num_extra_sentences = random.randint(min(len(evidences), 5), 5) - len(evidences)
                        extra_sentences = random.choices(candadates, k=num_extra_sentences)

                        datapoints['claim'].append(batch['claim'][i])
                        datapoints['evidences'].append(evidences)
                        datapoints['extra_sentences'].append(extra_sentences)
                        datapoints['label'].append(batch['label'][i])
                        datapoints['excluded_sentences'].append(None)

                    for _ in range(num_nei):
                        num_extra_sentences = random.randint(1, 5)
                        extra_sentences = random.choices(candadates, k=num_extra_sentences)
                        datapoints['claim'].append(batch['claim'][i])
                        datapoints['evidences'].append(None)
                        datapoints['extra_sentences'].append(extra_sentences)
                        datapoints['label'].append(2)
                        datapoints['excluded_sentences'].append(None)
            else:
                for eg in evidence_group:
                    evidences = set(sentence_mapping[(d, l)] for d, l in zip(eg['document'], eg['line']))
                    datapoints['claim'].append(batch['claim'][i])
                    datapoints['evidences'].append(evidences)
                    datapoints['extra_sentences'].append(None)
                    datapoints['label'].append(batch['label'][i])
                    datapoints['excluded_sentences'].append(sentence_ids)

                for _ in range(num_nei):
                    datapoints['claim'].append(batch['claim'][i])
                    datapoints['evidences'].append(None)
                    datapoints['extra_sentences'].append(None)
                    datapoints['label'].append(2)
                    datapoints['excluded_sentences'].append(sentence_ids)

        return datapoints

    dataset['train'] = dataset['train'].map(mapper, remove_columns=dataset['train'].column_names, batched=True, batch_size=1)
    dataset['test'] = dataset['test'].map(mapper, fn_kwargs=dict(is_test=True), remove_columns=dataset['test'].column_names, batched=True, batch_size=1)

    dataset.save_to_disk(output_dir)


if __name__ == '__main__':
    fire.Fire(main)
