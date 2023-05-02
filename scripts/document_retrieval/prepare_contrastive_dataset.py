import json
import random
from typing import Any, Dict, List

import fire
from datasets import disable_caching, load_from_disk
from tqdm.auto import tqdm


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
    full_document_ids = set()
    for i, x in enumerate(tqdm(wiki_dataset, desc='Creating Wiki Mapping', leave=False)):
        document_mapping[x['id']] = i
        full_document_ids.add(i)
    
    def mapper(batch: Dict[str, List[Any]], is_test: bool = False):
        datapoints = {
            'claim': [],
            'document': [],
            'label': [],
            'excluded_documents': [],
        }

        for i in range(len(batch['id'])):
            document_ids = set()
            if batch['label'][i] != 2:
                document_ids = set(document_mapping[n] for eg in batch['evidence'][i] for n in eg['document'])

                for di in document_ids:
                    datapoints['claim'].append(batch['claim'][i])
                    datapoints['document'].append(di)
                    datapoints['label'].append(1)
                    datapoints['excluded_documents'].append(None)

            num_negative = max(1, len(document_ids))
            if is_test:
                for di in random.choices(list(full_document_ids - document_ids), k=num_negative):
                    datapoints['claim'].append(batch['claim'][i])
                    datapoints['label'].append(-1)
                    datapoints['document'].append(di)
                    datapoints['excluded_documents'].append(None)
            else:
                for _ in range(num_negative):
                    datapoints['claim'].append(batch['claim'][i])
                    datapoints['label'].append(-1)
                    datapoints['document'].append(None)
                    datapoints['excluded_documents'].append(document_ids)

        return datapoints

    dataset['train'] = dataset['train'].map(mapper, remove_columns=dataset['train'].column_names, batched=True)
    dataset['test'] = dataset['test'].map(mapper, fn_kwargs=dict(is_test=True), remove_columns=dataset['test'].column_names, batched=True)

    dataset.save_to_disk(output_dir)

if __name__ == '__main__':
    fire.Fire(main)
