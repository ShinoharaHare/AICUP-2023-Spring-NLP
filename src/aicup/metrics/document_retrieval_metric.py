from typing import TypedDict

import torch
from torchmetrics import Metric


class DocumentRetrievalMetricOutput(TypedDict):
    accuracy: torch.FloatTensor
    precision: torch.FloatTensor
    recall: torch.FloatTensor
    f1: torch.FloatTensor


def adjusted_cosine_similarity(a: torch.Tensor, b: torch.Tensor):
    mean = (a.sum() + b.sum()) / a.size(1) / b.size(1)
    return torch.cosine_similarity(a - mean, b - mean)


class DocumentRetrievalMetric(Metric):
    tp: torch.LongTensor
    tn: torch.LongTensor
    fp: torch.LongTensor
    fn: torch.LongTensor

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()

        self.threshold = threshold

        self.add_state('tp', torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('tn', torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('fp', torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('fn', torch.tensor(0), dist_reduce_fx='sum')

    def update(
        self,
        embedding1: torch.FloatTensor,
        embedding2: torch.FloatTensor,
        label: torch.LongTensor
    ) -> None:
        cs = adjusted_cosine_similarity(embedding1, embedding2)
        positive = cs >= self.threshold
        ground_truth = label == 1

        self.tp += torch.count_nonzero(positive & ground_truth)
        self.tn += torch.count_nonzero(~positive & ~ground_truth)
        self.fp += torch.count_nonzero(positive & ~ground_truth)
        self.fn += torch.count_nonzero(~positive & ground_truth)
    
    def compute(self) -> DocumentRetrievalMetricOutput:
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn)
        precision = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.tp + self.fn)
        f1 = 2 / ((1 / precision) + (1 / recall))

        return DocumentRetrievalMetricOutput(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1
        )
