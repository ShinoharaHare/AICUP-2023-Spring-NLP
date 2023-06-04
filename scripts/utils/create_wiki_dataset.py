import os
from typing import Optional

import fire
from datasets import disable_caching

from aicup.data import WikiDataset


def main(
    data_dir: str,
    output_dir: str,
    num_proc: Optional[int] = None
):
    disable_caching()
    num_proc = num_proc or os.cpu_count()
    wiki_dataset = WikiDataset.from_json(data_dir, num_proc=num_proc)
    wiki_dataset.add_elasticsearch_index()
    wiki_dataset = wiki_dataset.save(output_dir)

if __name__ == '__main__':
    fire.Fire(main)
