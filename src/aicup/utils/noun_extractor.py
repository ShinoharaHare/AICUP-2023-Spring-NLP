from typing import List, Set

import hanlp
import hanlp.pretrained


class NounExtractor:
    def __init__(self) -> None:
        self.pipeline = (
            hanlp.pipeline()
            .append(
                hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH),
                output_key='tok',
            )
            .append(
                hanlp.load(hanlp.pretrained.pos.PKU_POS_ELECTRA_SMALL),
                input_key='tok',
                output_key='pos'
            )
        )

        self.valid_tags = {'n', 'nr', 'ns', 'nt', 'nx', 'nz', 't'}

    def merge_adjacent_nouns(self, tokens: List[str], tags: List[str]) -> Set[str]:
        i = 0
        merged_nouns = set()
        
        index_and_token = [(i, token) for i, (token, tag) in enumerate(zip(tokens, tags)) if tag in self.valid_tags]
        while i < len(index_and_token):
            ti, t = index_and_token[i]
            for ni in range(i + 1, len(index_and_token)):
                nti, nt = index_and_token[ni]
                if ti + 1 == nti:
                    t += nt
                    ti += 1
                else:
                    i = ni
                    break
            else:
                merged_nouns.add(t)
                break
            merged_nouns.add(t)
        return merged_nouns

    def extract(self, text: list[str], merge_adjacent: bool = False, return_unmerged: bool = False) -> list[set[str]]:
        results = self.pipeline(text)
        batch_nouns = []
        for tokens, tags in zip(results['tok'], results['pos']):
            nouns = set(token for token, tag in zip(tokens, tags) if tag in self.valid_tags)

            if merge_adjacent:
                merged_nouns = self.merge_adjacent_nouns(tokens, tags)
                if return_unmerged:
                    nouns |= merged_nouns
                else:
                    nouns = merged_nouns
            batch_nouns.append(nouns)
        return batch_nouns
