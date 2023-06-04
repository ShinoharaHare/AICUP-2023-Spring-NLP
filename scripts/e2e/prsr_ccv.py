import gc
import json
from pathlib import Path
from typing import Optional

import fire
import torch
from datasets import disable_caching, load_dataset

from aicup.data import WikiDataset
from aicup.models import (ClassifierClaimVerifier,
                          PairwiseRankingSentenceRetriever)
from aicup.utils import get_args


def free_memory():
    gc.collect()
    torch.cuda.empty_cache()

def main(
    wiki_dataset_path: str,
    test_data_path: str,
    sr_path: str,
    cv_path: str,
    output_path: str = 'predictions/e2e',
    precision: int = 16,
    device: str = 'cuda',
    sr_batch_size: int = 64,
    cv_batch_size: int = 32,
    sr_max_length: Optional[int] = None,
    cv_max_length: Optional[int] = None,
    top_k: int = 3,
    min_score: float = 10.0,
    return_by_noun: bool = True,
    merge_adjacent: bool = True,
    return_unmerged: bool = True,
):
    config = get_args()
    disable_caching()

    dtype = {16: torch.half, 32: torch.float}[precision]
    device = torch.device(device)
    wiki_dataset = WikiDataset.load(wiki_dataset_path)
    wiki_dataset.load_elasticsearch_index()
    dataset = load_dataset('json', data_files=test_data_path)['train']
    # dataset = dataset.select(range(32))

    def retrieve_document(batch: dict[str, list]):
        new_batch = {
            'id': [],
            'claim': [],
            'title': [],
            'sentence': [],
            'predicted_evidence_title': [],
            'predicted_evidence_line': [],
        }

        batch_retrieved_documents = wiki_dataset.retrieve(
            batch['claim'],
            top_k=top_k,
            min_score=min_score,
            return_by_noun=return_by_noun,
            merge_adjacent=merge_adjacent,
            return_unmerged=return_unmerged,
        )

        for id_, claim, retrieved_document, retrieved_documents in zip(batch['id'], batch['claim'], batch_retrieved_documents, batch_retrieved_documents):
            for title, sentences in zip(retrieved_document['id'], retrieved_documents['lines']):
                for line, sentence in enumerate(sentences):
                    if not sentence:
                        continue

                    new_batch['id'].append(id_)
                    new_batch['claim'].append(claim)
                    new_batch['title'].append(title)
                    new_batch['sentence'].append(sentence)
                    new_batch['predicted_evidence_title'].append(title)
                    new_batch['predicted_evidence_line'].append(line)

        return new_batch

    dataset = dataset.map(
        retrieve_document,
        remove_columns=dataset.column_names,
        batched=True,
    )

    dataset.save_to_disk('temp/1')

    del wiki_dataset
    free_memory()

    sr = PairwiseRankingSentenceRetriever.load_from_checkpoint(sr_path, device)
    sr = sr.eval().to(dtype=dtype, device=device)

    def score_sentences(batch: dict[str, list]):
        new_batch = {}
        batch_score = sr.predict(
            batch['claim'],
            batch['title'],
            batch['sentence'],
            max_length=sr_max_length,
        )
        new_batch['score'] = batch_score
        return new_batch

    dataset = dataset.map(
        score_sentences,
        batched=True,
        batch_size=sr_batch_size,
    )
    dataset.save_to_disk('temp/2')

    del sr
    free_memory()
    
    def aggregate_sentences(batch: dict[str, list]):
        new_batch = {
            'id': [],
            'claim': [],
            'titles': [],
            'sentences': [],
            'predicted_evidence_titles': [],
            'predicted_evidence_lines': [],
        }

        mapping: dict[str, list] = {}
        for i, id_ in enumerate(batch['id']):
            l = mapping.setdefault(id_, [])
            l.append({k: v[i] for k, v in batch.items()})

        for k, v in mapping.items():
            v.sort(key=lambda x: x['score'], reverse=True)

            titles = []
            sentences = []
            predicted_evidence_titles = []
            predicted_evidence_lines = []

            for y in v[:5]:
                titles.append(y['title'])
                sentences.append(y['sentence'])
                predicted_evidence_titles.append(y['predicted_evidence_title'])
                predicted_evidence_lines.append(y['predicted_evidence_line'])
            
            new_batch['id'].append(k)
            new_batch['claim'].append(v[0]['claim'])
            new_batch['titles'].append(titles)
            new_batch['sentences'].append(sentences)
            new_batch['predicted_evidence_titles'].append(predicted_evidence_titles)
            new_batch['predicted_evidence_lines'].append(predicted_evidence_lines)

        return new_batch

    dataset = dataset.map(aggregate_sentences, batched=True, batch_size=dataset.num_rows, remove_columns=dataset.column_names)
    dataset.save_to_disk('temp/3')

    cv = ClassifierClaimVerifier.load_from_checkpoint(cv_path, device)
    cv = cv.eval().to(dtype=dtype, device=device)

    def predict_label(batch: dict[str, list]):
        new_batch = {
            'predicted_label': cv.predict(
                batch['claim'],
                batch['titles'],
                batch['sentences'],
                max_length=cv_max_length,
            ).tolist(),
        }        
        return new_batch
    dataset = dataset.map(predict_label, batched=True, batch_size=cv_batch_size)

    del cv
    free_memory()

    output_path: Path = Path(output_path)
    n = len([p for p in output_path.glob('*') if p.is_dir()])
    output_path = output_path.joinpath(f'{n}')
    output_path.mkdir(parents=True, exist_ok=False)

    label_mapping = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
    with open(output_path.joinpath('submission.jsonl'), 'w', encoding='utf-8') as f:
        for x in dataset:
            predicted_evidence = None
            if x['predicted_label'] != 2:
                predicted_evidence = [[title, line] for title, line in zip(x['predicted_evidence_titles'], x['predicted_evidence_lines'])]

            f.write(json.dumps({
                'id': x['id'],
                'predicted_label': label_mapping[x['predicted_label']],
                'predicted_evidence': predicted_evidence,
            }, ensure_ascii=False))
            f.write('\n')
    
    with open(output_path.joinpath('config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=True)

if __name__ == '__main__':
    fire.Fire(main)
