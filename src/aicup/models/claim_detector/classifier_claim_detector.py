from typing import TypedDict

import torch
from bitsandbytes.optim import AdamW8bit
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from torch import LongTensor, nn
from torch.optim import AdamW
from torchmetrics import Accuracy, F1Score
from transformers import LlamaModel, LlamaTokenizerFast

from ...utils.training import (LightningModuleX, PartiallyFrozenEmbedding,
                               get_warmup_scheduler)
from ..classifier import LLaMAClassifier


class _BatchType(TypedDict):
    input_ids: LongTensor
    attention_mask: LongTensor
    label: LongTensor


class ClassifierClaimDetector(LightningModuleX):

    def __init__(self) -> None:
        super().__init__()

        self.model_path = 'tmpfs/llama-7b-extended_stage1--b-2048--e1_stage2--b-2048--e1'
        self.tokenizer: LlamaTokenizerFast = LlamaTokenizerFast.from_pretrained(self.model_path)
        self.tokenizer.add_special_tokens({'sep_token': '<sep>'})
        self.loss_fn = nn.CrossEntropyLoss()

        self.train_accuracy = Accuracy(task='multiclass', num_classes=3)
        self.train_f1 = F1Score(task='multiclass', num_classes=3)

        self.val_accuracy = Accuracy(task='multiclass', num_classes=3)
        self.val_f1 = F1Score(task='multiclass', num_classes=3)

    def configure_sharded_model(self) -> None:
        if hasattr(self, 'encoder'):
            return
        
        llama: LlamaModel = LlamaModel.from_pretrained(
            self.model_path,
            device_map={'': 0},
            low_cpu_mem_usage=True,
            torch_dtype=torch.half,
            load_in_8bit=True,
        )
        old_num_tokens = llama.embed_tokens.num_embeddings
        new_num_tokens = len(self.tokenizer)
        llama.resize_token_embeddings(new_num_tokens)
        input_embeddings = llama.get_input_embeddings()
        new_embedding_weight = input_embeddings.weight[:new_num_tokens].mean(0)
        input_embeddings.weight.data[new_num_tokens:].copy_(new_embedding_weight)
        llama = prepare_model_for_int8_training(llama)
        input_embeddings.float().requires_grad_()

        lora_config = LoraConfig(
            r=8,
            target_modules=['q_proj', 'v_proj'],
            lora_alpha=16,
            lora_dropout=0.05
        )
        llama = get_peft_model(llama, lora_config)
        
        self.classifier = LLaMAClassifier(llama)

    def configure_optimizers(self):
        optimizer_config = {}
        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer_config['optimizer'] = AdamW8bit(parameters, lr=1e-4)
        optimizer_config['lr_scheduler'] = {
            'scheduler': get_warmup_scheduler(
                optimizer_config['optimizer'],
                num_warmup_steps=500,
            ),
            'interval': 'step',
        }
        return optimizer_config

    def training_step(self, batch: _BatchType, batch_idx: int):
        batch_size = batch['input_ids'].size(0)
        x = self.classifier.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            use_cache=False,
        )
        loss = self.loss_fn(x, batch['label'])
        
        self.log('global_step', self.global_step + 1, prog_bar=True, logger=False)
        self.log('loss', loss, prog_bar=True, logger=False)
        self.log('Loss/Train', loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('Loss/Train/Step', loss)

        self.train_accuracy.update(x, batch['label'])
        self.train_f1.update(x, batch['label'])
        return loss

    def on_train_epoch_end(self) -> None:
        self.log('Accuracy/Train', self.train_accuracy, sync_dist=True)
        self.log('F1/Train', self.train_f1, sync_dist=True)

    def validation_step(self, batch: _BatchType, batch_idx: int):
        batch_size = batch['input_ids'].size(0)
        x = self.classifier.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
        )
        loss = self.loss_fn(x, batch['label'])

        self.log('loss', loss, prog_bar=True, logger=False, sync_dist=True)
        self.log('Loss/Val', loss, sync_dist=True, batch_size=batch_size)

        self.val_accuracy.update(x, batch['label'])
        self.val_f1.update(x, batch['label'])

        self.log('Accuracy/Val', self.val_accuracy, sync_dist=True, batch_size=batch_size)
        self.log('F1/Val', self.val_f1, sync_dist=True, batch_size=batch_size)
        return loss
