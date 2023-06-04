import json
import os

import fire
from datasets import disable_caching, load_from_disk

from aicup.data import WikiDataset
from aicup.utils import get_args


def main(
    claim_dataset_path: str,
    wiki_dataset_path: str,
    output_dir: str,
    include_retrieved: bool = False,
    top_k: int = 3,
    min_score: float = 10.0,
    return_by_noun: bool = True,
    merge_adjacent: bool = True,
    return_unmerged: bool = True,
):
    config = get_args()
    disable_caching()

    claim_dataset = load_from_disk(claim_dataset_path)
    wiki_dataset = WikiDataset.load(wiki_dataset_path)
    wiki_dataset.load_elasticsearch_index()
    
    def mapper(batch: dict[str, list]):
        new_batch = {
            'claim': [],
            'sentences': [],
            'is_evidence': []
        }

        evidence_documents = []
        for eg in batch['evidence']:
            eds = set()
            eg = eg or []
            for e in eg:
                for dt in e['document']:
                    eds.add(dt)
            evidence_documents.append(eds)
        
        retrieved_documents = []
        if include_retrieved:
            batch_examples = wiki_dataset.retrieve(
                batch['claim'],
                top_k=top_k,
                min_score=min_score,
                return_by_noun=return_by_noun,
                merge_adjacent=merge_adjacent,
                return_unmerged=return_unmerged,
            )
            for examples in batch_examples:
                retrieved_documents.append(set(examples['id']))
        else:
            retrieved_documents = [set()] * len(evidence_documents)

        for claim, eg, eds, rds in zip(batch['claim'], batch['evidence'], evidence_documents, retrieved_documents):
            eg = eg or []
            evidence_sentences = set()
            for e in eg:
                for dt, li in zip(e['document'], e['line']):
                    evidence_sentences.add((dt, li))

            document_candidates = eds | rds
            for dt in document_candidates:
                sentences = []
                is_evidence = []

                for li, l in enumerate(wiki_dataset[dt]['lines']):
                    sentences.append(l)
                    is_evidence.append((dt, li) in evidence_sentences)

                new_batch['claim'].append(claim)
                new_batch['sentences'].append(sentences)
                new_batch['is_evidence'].append(is_evidence)

        return new_batch

    claim_dataset = claim_dataset.map(
        mapper,
        remove_columns=claim_dataset['train'].column_names,
        batched=True,
    )

    claim_dataset.save_to_disk(output_dir)

    with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=True)

if __name__ == '__main__':
    fire.Fire(main)
