from typing import Dict

import lightning as L
import torch
from bitsandbytes.optim import AdamW8bit
from torch import Tensor, nn
from torch.optim import AdamW
from torchmetrics import Accuracy, F1Score, Precision, Recall
from transformers import BertTokenizerFast, DebertaV2ForSequenceClassification

from ...utils.training import get_warmup_scheduler

class ClassifierDocumentRetreiver(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()

        self.num_labels = 1187751

        self.tokenizer = BertTokenizerFast.from_pretrained('IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese')

        self.val_metrics = nn.ModuleDict({
            'accuracy': Accuracy(task='multilabel', num_labels=self.num_labels, top_k=5),
            'f1_score': F1Score(task='multilabel', num_labels=self.num_labels, top_k=5),
            'precision': Precision(task='multilabel', num_labels=self.num_labels, top_k=5),
            'recall': Recall(task='multilabel', num_labels=self.num_labels, top_k=5),
        })

        self.train_metrics = nn.ModuleDict({
            'accuracy': Accuracy(task='multilabel', num_labels=self.num_labels, top_k=5),
            'f1_score': F1Score(task='multilabel', num_labels=self.num_labels, top_k=5),
            'precision': Precision(task='multilabel', num_labels=self.num_labels, top_k=5),
            'recall': Recall(task='multilabel', num_labels=self.num_labels, top_k=5),
        })

    def configure_sharded_model(self) -> None:
        self.deberta = DebertaV2ForSequenceClassification.from_pretrained(
            'IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese',
            problem_type='multi_label_classification',
            num_labels=self.num_labels,
            # torch_dtype=torch.half,
        )
        self.deberta.gradient_checkpointing_enable()
        # self.deberta = torch.compile(self.deberta)

    def configure_optimizers(self):
        optimizer_config = {}
        optimizer_config['optimizer'] = AdamW8bit([p for p in self.parameters() if p.requires_grad], lr=1e-5)
        optimizer_config['lr_scheduler'] = {
            'scheduler': get_warmup_scheduler(
                optimizer_config['optimizer'],
                num_warmup_steps=100,
            ),
            'interval': 'step',
        }
        return optimizer_config 

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int):
        output = self.deberta.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = output.loss

        metrics = self.train_metrics
        target = batch['labels'] > 0.5
        target = target.long()

        self.log('Loss/Train', loss)
        self.log('Accuracy/Train', metrics['accuracy'](output.logits, target))
        self.log('F1Score/Train', metrics['f1_score'](output.logits, target))
        self.log('Precision/Train', metrics['precision'](output.logits, target))
        self.log('Recall/Train', metrics['recall'](output.logits, target))
        return loss
    
    def on_validation_start(self) -> None:
        self.deberta.gradient_checkpointing_enable()

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int):
        output = self.deberta.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = output.loss

        metrics = self.val_metrics
        target = batch['labels'] > 0.5
        target = target.long()
        metrics['accuracy'].update(output.logits, target)
        metrics['f1_score'].update(output.logits, target)
        metrics['precision'].update(output.logits, target)
        metrics['recall'].update(output.logits, target)

        self.log('Loss/Val', loss)
        self.log('Accuracy/Val', metrics['accuracy'])
        self.log('F1Score/Val', metrics['f1_score'])
        self.log('Precision/Val', metrics['precision'])
        self.log('Recall/Val', metrics['recall'])

    def on_validation_end(self) -> None:
        self.deberta.gradient_checkpointing_disable()
