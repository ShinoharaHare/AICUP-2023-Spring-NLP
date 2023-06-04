import inspect
import json
import os

import fire
from datasets import disable_caching, load_from_disk

from aicup.data import WikiDataset
from aicup.models import TokenLevelSentenceRetriever
from aicup.utils import get_args


def main(
    claim_dataset_path: str,
    wiki_dataset_path: str,
    output_dir: str,
    sentence_retriever_path: str,
    top_k: int = 10,
    min_score: float = 0.0,
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

    sentence_retriever = TokenLevelSentenceRetriever.load_from_checkpoint(sentence_retriever_path)
    sentence_retriever = sentence_retriever.cuda()
    
    def mapper(batch: dict[str, list]):
        new_batch = {
            'claim': [],
            'sentences': [],
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
        for claim, retrieved_documents, label in zip(batch['claim'], batch_retrieved_documents, batch['label']):
            if not retrieved_documents['id']:
                continue
            batch_scores = sentence_retriever.predict([claim] * len(retrieved_documents['id']), retrieved_documents['lines'])

            score_with_index = []
            for i, scores in enumerate(batch_scores):
                for j, score in enumerate(scores):
                    score_with_index.append((i, j, score.item()))
            score_with_index.sort(key=lambda x: x[2], reverse=True)
            sentences = [retrieved_documents['lines'][di][si] for di, si, _ in score_with_index[:5]]
            new_batch['claim'].append(claim)
            new_batch['sentences'].append(sentences)
            new_batch['label'].append(label)
        return new_batch

    dataset = claim_dataset.map(
        mapper,
        remove_columns=claim_dataset['train'].column_names,
        batched=True,
        batch_size=64,
        # num_proc=4,
    )

    dataset.save_to_disk(output_dir)

    with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=True)


if __name__ == '__main__':
    fire.Fire(main)
