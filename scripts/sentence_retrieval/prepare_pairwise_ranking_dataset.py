import json
import os
import random

import fire
from datasets import disable_caching, load_from_disk

from aicup.data import WikiDataset
from aicup.utils import get_args


def main(
    claim_dataset_path: str,
    wiki_dataset_path: str,
    output_dir: str,
    top_k: int = 3,
    min_score: float = 10.0,
    return_by_noun: bool = True,
    merge_adjacent: bool = True,
    return_unmerged: bool = True,
    seed: int = 42,
):
    config = get_args()
    random.seed(seed)
    disable_caching()

    claim_dataset = load_from_disk(claim_dataset_path)
    wiki_dataset = WikiDataset.load(wiki_dataset_path)
    wiki_dataset.load_elasticsearch_index()
    
    def mapper(batch: dict[str, list]):
        new_batch = {
            'positive_claim': [],
            'positive_sentence': [],
            'negative_claim': [],
            'negative_sentence': []
        }

        batch_evidence_sentences = []
        for eg in batch['evidence']:
            eds = set()
            eg = eg or []
            for e in eg:
                for dt, li in zip(e['document'], e['line']):
                    eds.add((dt, wiki_dataset[dt]['lines'][li]))
            batch_evidence_sentences.append(eds)
        
        batch_examples = wiki_dataset.retrieve(
            batch['claim'],
            top_k=top_k,
            min_score=min_score,
            return_by_noun=return_by_noun,
            merge_adjacent=merge_adjacent,
            return_unmerged=return_unmerged,
        )
        batch_retrieved_sentences = [set((dt, s) for s in lines) for examples in batch_examples for dt, lines in zip(examples['id'], examples['lines'])]

        for claim, evidence_sentences, retrieved_sentences in zip(batch['claim'], batch_evidence_sentences, batch_retrieved_sentences):
            if not evidence_sentences:
                for negative_sentence in retrieved_sentences:
                    if not negative_sentence:
                        continue

                    while True:
                        i = random.choice(list(range(len(batch['claim']))))
                        if batch_evidence_sentences[i]:
                            positive_claim = batch['claim'][i]
                            positive_sentence = random.choice(list(batch_evidence_sentences[i]))
                            break

                    new_batch['positive_claim'].append(positive_claim)
                    new_batch['positive_sentence'].append(positive_sentence)

                    new_batch['negative_claim'].append(claim)
                    new_batch['negative_sentence'].append(negative_sentence)
                continue
            
            negative_sentences = retrieved_sentences - evidence_sentences
            for positive_sentence in evidence_sentences:
                for negative_sentence in negative_sentences:
                    if not negative_sentence:
                        continue

                    new_batch['positive_claim'].append(claim)
                    new_batch['positive_sentence'].append(positive_sentence)
                    new_batch['negative_claim'].append(claim)
                    new_batch['negative_sentence'].append(negative_sentence)

        return new_batch

    dataset = claim_dataset.map(
        mapper,
        remove_columns=claim_dataset['train'].column_names,
        batched=True,
    )

    dataset.save_to_disk(output_dir)

    with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=True)

if __name__ == '__main__':
    fire.Fire(main)
