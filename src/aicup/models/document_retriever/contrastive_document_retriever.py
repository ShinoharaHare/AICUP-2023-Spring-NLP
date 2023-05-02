from typing import Dict, TypedDict, Union

import torch
from bitsandbytes.optim import AdamW8bit
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from torch import LongTensor, Tensor, nn
from torch.optim import AdamW
from transformers import BatchEncoding, LlamaModel, LlamaTokenizerFast

from ...metrics import DocumentRetrievalMetric
from ...utils.training import LightningModuleX, get_warmup_scheduler
from ..text_embedder import LLaMATextEmbedder


class _BatchType(TypedDict):
    claim: BatchEncoding
    document: BatchEncoding
    label: LongTensor


class ContrastiveDocumentRetriever(LightningModuleX):

    def __init__(self) -> None:
        super().__init__()

        self.model_path = 'tmpfs/llama-7b-extended_stage1--b-2048--e1_stage2--b-2048--e1'
        self.tokenizer = LlamaTokenizerFast.from_pretrained(self.model_path)
        self.loss_fn = nn.CosineEmbeddingLoss()

        self.train_dr_metric = DocumentRetrievalMetric()
        self.val_dr_metric = DocumentRetrievalMetric()
        self.batch_dr_metric = DocumentRetrievalMetric()

    def configure_sharded_model(self) -> None:
        if hasattr(self, 'embedder'):
            return
        
        llama = LlamaModel.from_pretrained(
            self.model_path,
            device_map={'': 0},
            low_cpu_mem_usage=True,
            torch_dtype=torch.half,
            load_in_8bit=True,
        )
        llama = prepare_model_for_int8_training(llama)
        lora_config = LoraConfig(
            r=8,
            target_modules=['q_proj', 'v_proj'],
            lora_alpha=16,
            lora_dropout=0.05
        )
        llama = get_peft_model(llama, lora_config)
        self.embedder = LLaMATextEmbedder(llama)

    def configure_optimizers(self):
        optimizer_config = {}
        optimizer_config['optimizer'] = AdamW8bit([p for p in self.parameters() if p.requires_grad], lr=1e-4)
        optimizer_config['lr_scheduler'] = {
            'scheduler': get_warmup_scheduler(
                optimizer_config['optimizer'],
                num_warmup_steps=200,
            ),
            'interval': 'step',
        }
        return optimizer_config

    def training_step(self, batch: _BatchType, batch_idx: int):
        batch_size = batch['label'].size(0)
        embedding1 = self.embedder.forward(**batch['claim'], use_cache=False)
        embedding2 = self.embedder.forward(**batch['document'], use_cache=False)
        loss = self.loss_fn(embedding1, embedding2, batch['label'])

        self.log('global_step', self.global_step + 1, prog_bar=True, logger=False)
        self.log('loss', loss, prog_bar=True, logger=False)
        self.log('Loss/Train', loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('Loss/Train/Step', loss)

        self.train_dr_metric.update(embedding1, embedding2, batch['label'])
        self.batch_dr_metric.update(embedding1, embedding2, batch['label'])
        metric_output = self.batch_dr_metric.compute()
        self.batch_dr_metric.reset()
        self.log_dict({
            'Accuracy/Train/Step': metric_output['accuracy'],
            'Precision/Train/Step': metric_output['precision'],
            'Recall/Train/Step': metric_output['recall'],
            'F1/Train/Step': metric_output['f1'],
        })
        return loss

    def on_train_epoch_end(self) -> None:
        metric_output = self.train_dr_metric.compute()
        self.train_dr_metric.reset()
        self.log_dict(
            {
                'Accuracy/Train': metric_output['accuracy'],
                'Precision/Train': metric_output['precision'],
                'Recall/Train': metric_output['recall'],
                'F1/Train': metric_output['f1'],
            },
            sync_dist=True
        )

    def validation_step(self, batch: _BatchType, batch_idx: int):
        batch_size = batch['label'].size(0)
        embedding1 = self.embedder.forward(**batch['claim'])
        embedding2 = self.embedder.forward(**batch['document'])
        loss = self.loss_fn(embedding1, embedding2, batch['label'])

        self.log('loss', loss, prog_bar=True, logger=False, sync_dist=True, batch_size=batch_size)
        self.log('Loss/Val', loss, sync_dist=True, batch_size=batch_size)

        self.val_dr_metric.update(embedding1, embedding2, batch['label'])
        return loss

    def on_validation_epoch_end(self) -> None:
        metric_output = self.val_dr_metric.compute()
        self.val_dr_metric.reset()
        self.log_dict(
            {
                'Accuracy/Val': metric_output['accuracy'],
                'Precision/Val': metric_output['precision'],
                'Recall/Val': metric_output['recall'],
                'F1/Val': metric_output['f1'],
            },
            sync_dist=True
        )
