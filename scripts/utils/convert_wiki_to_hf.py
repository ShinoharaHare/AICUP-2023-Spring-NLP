import os
import re
from typing import Dict, Optional

import fire
from datasets import disable_caching, load_dataset


def main(
    data_dir: str,
    output_dir: str,
    num_proc: Optional[int] = None
):
    disable_caching()
    num_proc = num_proc or os.cpu_count()
    wiki_dataset = load_dataset('json', data_dir=data_dir)['train']
    
    def mapper(x: Dict[str, str]):
        x['raw_lines'] = x['lines']
        x['lines'] = re.split(r'^\d+\t', x['raw_lines'], flags=re.MULTILINE)[1:]
        x['lines'] = [l.strip() for l in x['lines']]

        # i = 0
        # while x['raw_lines']:
        #     st = f'{i}\t'
        #     et = f'{i + 1}\t'
        #     si = x['raw_lines'].find(st) + len(st)
        #     ei = x['raw_lines'].find(et, si)
            
        #     if ei == -1:
        #         x['lines'].append(x['raw_lines'][si:].strip())
        #         break
        #     else:
        #         x['lines'].append(x['raw_lines'][si:ei].strip())
        #         i += 1

        return x
    
    wiki_dataset = wiki_dataset.map(mapper, num_proc=num_proc)
    wiki_dataset.save_to_disk(output_dir)

if __name__ == '__main__':
    fire.Fire(main)
