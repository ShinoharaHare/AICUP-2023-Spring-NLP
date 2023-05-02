import re

from datasets import load_from_disk
from tqdm.auto import tqdm

wiki_dataset = load_from_disk('data/wiki')
for x in tqdm(wiki_dataset):
    if not x['raw_lines'] and not x['lines']:
        continue
    m = re.search(r'(\d+)\t.*$', x['raw_lines'])
    assert len(x['lines']) - 1 == int(m.group(1))
    for i, l in enumerate(x['lines']):
        assert re.search(rf'{i}\t\s*{re.escape(l)}\s*', x['raw_lines']) is not None
