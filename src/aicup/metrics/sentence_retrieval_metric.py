from typing import TypedDict

import torch
from torchmetrics import Metric


class SentenceRetrievalMetricOutput(TypedDict):
    accuracy: torch.FloatTensor
    precision: torch.FloatTensor
    recall: torch.FloatTensor
    f1: torch.FloatTensor


class SentenceRetrievalMetric(Metric):
    tp: torch.LongTensor
    tn: torch.LongTensor
    fp: torch.LongTensor
    fn: torch.LongTensor

    def __init__(self, threshold: float = 0.8) -> None:
        super().__init__()

        self.threshold = threshold

        self.add_state('tp', torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('tn', torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('fp', torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('fn', torch.tensor(0), dist_reduce_fx='sum')

    def update(
        self,
        preds: torch.FloatTensor,
        score: torch.FloatTensor,
    ) -> None:
        positive = preds >= self.threshold
        ground_truth = score >= self.threshold

        self.tp += torch.count_nonzero(positive & ground_truth)
        self.tn += torch.count_nonzero(~positive & ~ground_truth)
        self.fp += torch.count_nonzero(positive & ~ground_truth)
        self.fn += torch.count_nonzero(~positive & ground_truth)
    
    def compute(self) -> SentenceRetrievalMetricOutput:
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn)
        precision = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.tp + self.fn)
        f1 = 2 / ((1 / precision) + (1 / recall))

        return SentenceRetrievalMetricOutput(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1
        )
