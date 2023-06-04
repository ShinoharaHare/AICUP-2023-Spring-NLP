import json
from typing import List

import fire
from datasets import (ClassLabel, Dataset, Features, Sequence, Value,
                      disable_caching)


def generator(data_files: List[str]):
    evidence_attr_names = ['annotation_id', 'evidence_id', 'document', 'line']
    for data_file in data_files:
        with open(data_file, encoding='utf-8') as f:
            for l in f:
                x = json.loads(l)
                x['label'] = x['label'].upper()
                if x['label'] == 'NOT ENOUGH INFO':
                    x['evidence'] = None
                else:
                    for eg in x['evidence']:
                        for i, e in enumerate(eg):
                            eg[i] = {}
                            for attr_name, attr in zip(evidence_attr_names, e):
                                if attr_name == 'document' and attr == '臺灣海峽危機#第二次臺灣海峽危機（1958）':
                                    attr = '臺灣海峽危機'
                                eg[i][attr_name] = attr
                yield x

def main(
    data_files: List[str],
    output_dir: str,
    seed: int = 42
):
    disable_caching()

    dataset = Dataset.from_generator(
        generator,
        gen_kwargs=dict(data_files=data_files),
        features=Features({
            'id': Value('int32'),
            'label': ClassLabel(3, names=['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']),
            'claim': Value('string'),
            'evidence': Sequence(Sequence({
                'annotation_id': Value('int32'),
                'evidence_id': Value('int32'),
                'document': Value('string'),
                'line': Value('int32'),
            }))
        })
    )
    dataset = dataset.train_test_split(0.1, seed=seed)
    dataset.save_to_disk(output_dir)


if __name__ == '__main__':
    fire.Fire(main)
