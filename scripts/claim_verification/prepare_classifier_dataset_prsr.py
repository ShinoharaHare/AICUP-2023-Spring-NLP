import json
import os

import fire
from datasets import Dataset, DatasetDict, disable_caching, load_from_disk

from aicup.data import WikiDataset
from aicup.models.sentence_retriever import PairwiseRankingSentenceRetriever
from aicup.utils import get_args


def main(
    claim_dataset_path: str,
    wiki_dataset_path: str,
    output_dir: str,
    sentence_retriever_path: str,
    top_k: int = 3,
    min_score: float = 10.0,
    return_by_noun: bool = True,
    merge_adjacent: bool = True,
    return_unmerged: bool = True,
):
    config = get_args()
    disable_caching()

    claim_dataset = load_from_disk(claim_dataset_path)

    # claim_dataset['train'] = claim_dataset['train'].select(range(128))
    # claim_dataset['test'] = claim_dataset['test'].select(range(128))

    wiki_dataset = WikiDataset.load(wiki_dataset_path)
    wiki_dataset.load_elasticsearch_index()
    
    def retrieve_document(batch: dict[str, list]):
        new_batch = {
            'claim': [],
            'title': [],
            'sentence': [],
            'label': []
        }

        batch_retrieved_documents = wiki_dataset.retrieve(
            batch['claim'],
            top_k=top_k,
            min_score=min_score,
            return_by_noun=return_by_noun,
            merge_adjacent=merge_adjacent,
            return_unmerged=return_unmerged,
        )

        for i in range(len(batch['id'])):
            for title, sentences in zip(batch_retrieved_documents[i]['id'], batch_retrieved_documents[i]['lines']):
                for sentence in sentences:
                    if not sentence:
                        continue

                    new_batch['claim'].append(batch['claim'][i])
                    new_batch['title'].append(title)
                    new_batch['sentence'].append(sentence)
                    new_batch['label'].append(batch['label'][i])

        return new_batch

    dataset = claim_dataset.map(
        retrieve_document,
        remove_columns=claim_dataset['train'].column_names,
        batched=True,
    )

    prsr = PairwiseRankingSentenceRetriever.load_from_checkpoint(sentence_retriever_path, 'cuda')
    prsr = prsr.half().cuda()

    def retrieve_sentence(batch: dict[str, list]):
        new_batch = {}
        batch_score = prsr.predict(batch['claim'], batch['title'], batch['sentence'])
        new_batch['score'] = batch_score
        return new_batch

    sr_result = dataset.map(
        retrieve_sentence,
        batched=True,
        batch_size=64
    )

    dataset = DatasetDict()
    for split, result in sr_result.items():
        mapping: dict[str, list] = {}
        for x in result:
            l = mapping.setdefault(x['claim'], [])
            l.append(x)

        l = []
        for k, v in mapping.items():
            v.sort(key=lambda x: x['score'], reverse=True)
            l.append({
                'claim': k,
                'titles': [x['title'] for x in v[:5]],
                'sentences': [x['sentence'] for x in v[:5]],
                'label': v[0]['label']
            })
        dataset[split] = Dataset.from_list(l)
    
    dataset.save_to_disk(output_dir)

    with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=True)


if __name__ == '__main__':
    fire.Fire(main)
