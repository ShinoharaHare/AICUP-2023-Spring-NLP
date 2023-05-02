import json

import fire
from datasets import (ClassLabel, Dataset, Features, Sequence, Value,
                      disable_caching)


def generator(data_path: str):
    evidence_attr_names = ['annotation_id', 'evidence_id', 'document', 'line']

    with open(data_path, encoding='utf-8') as f:
        for l in f:
            x = json.loads(l)
            if x['label'] == 'NOT ENOUGH INFO':
                x['evidence'] = [x['evidence']]
            for eg in x['evidence']:
                for i, e in enumerate(eg):
                    eg[i] = {evidence_attr_names[i]: x for i, x in enumerate(e)}
            yield x

def main(
    data_path: str,
    output_dir: str,
):
    disable_caching()

    dataset = Dataset.from_generator(
        generator,
        gen_kwargs=dict(data_path=data_path),
        features=Features({
            'id': Value('int32'),
            'label': ClassLabel(3, names=['supports', 'refutes', 'NOT ENOUGH INFO']),
            'claim': Value('string'),
            'evidence': Sequence(Sequence({
                'annotation_id': Value('int32'),
                'evidence_id': Value('int32'),
                'document': Value('string'),
                'line': Value('int32'),
            }))
        })
    )
    dataset = dataset.train_test_split(0.1, seed=42)
    dataset.save_to_disk(output_dir)


if __name__ == '__main__':
    fire.Fire(main)
