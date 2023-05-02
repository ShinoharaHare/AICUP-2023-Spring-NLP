import gc
from typing import Any, Dict, TypedDict

import torch
from bitsandbytes.optim import AdamW8bit
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from torch import FloatTensor, LongTensor, nn
from torch.optim import AdamW
from transformers import LlamaModel, LlamaTokenizerFast

from ...metrics import SentenceRetrievalMetric
from ...utils.training import (LightningModuleX, PartiallyFrozenEmbedding,
                               get_warmup_scheduler)
from ..cross_encoder import LLaMACrossEncoder


class _BatchType(TypedDict):
    input_ids: LongTensor
    attention_mask: LongTensor
    score: FloatTensor


class CrossEncoderSentenceRetriever(LightningModuleX):

    def __init__(self) -> None:
        super().__init__()

        # self.model_path = 'tmpfs/llama-7b-extended_stage1--b-2048--e1_stage2--b-2048--s12620'
        self.model_path = 'tmpfs/llama-7b-extended_stage1--b-2048--e1_stage2--b-2048--e1'
        self.tokenizer: LlamaTokenizerFast = LlamaTokenizerFast.from_pretrained(self.model_path)
        self.tokenizer.add_special_tokens({'sep_token': '<sep>'})
        self.loss_fn = nn.BCEWithLogitsLoss()

        # self.learning_rate = 1e-4
        # self.betas = (0.9, 0.999)
        # self.weight_decay = 1e-2
        # self.num_warmup_steps = 200

        self.learning_rate = 5e-5
        self.betas = (0.9, 0.95)
        self.weight_decay = 1e-1
        self.num_warmup_steps = 500

        self.train_sr_metric = SentenceRetrievalMetric()
        self.val_sr_metric = SentenceRetrievalMetric()
        self.batch_sr_metric = SentenceRetrievalMetric()

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
        
        # embedding = PartiallyFrozenEmbedding(llama.embed_tokens, old_num_tokens)
        # llama.set_input_embeddings(embedding)
        llama = prepare_model_for_int8_training(llama)
        lora_config = LoraConfig(
            r=8,
            target_modules=['q_proj', 'v_proj'],
            lora_alpha=16,
            lora_dropout=0.05
        )
        llama = get_peft_model(llama, lora_config)
        # embedding.w2.data = embedding.w2.float()
        # embedding.w2.requires_grad_()
        input_embeddings.float().requires_grad_()
        self.encoder = LLaMACrossEncoder(llama)
    
    def configure_optimizers(self):
        optimizer_config = {}
        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer_config['optimizer'] = AdamW8bit(parameters, lr=self.learning_rate, betas=self.betas, weight_decay=self.weight_decay)
        # optimizer_config['optimizer'] = AdamW([p for p in self.parameters() if p.requires_grad], lr=1e-5)
        optimizer_config['lr_scheduler'] = {
            'scheduler': get_warmup_scheduler(
                optimizer_config['optimizer'],
                num_warmup_steps=self.num_warmup_steps,
            ),
            'interval': 'step',
        }
        return optimizer_config

    def training_step(self, batch: _BatchType, batch_idx: int):
        batch_size = batch['input_ids'].size(0)
        x = self.encoder.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            use_cache=False,
        )
        loss = self.loss_fn(x, batch['score'])
        
        self.log('global_step', self.global_step + 1, prog_bar=True, logger=False)
        self.log('loss', loss, prog_bar=True, logger=False)
        self.log('Loss/Train', loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('Loss/Train/Step', loss)

        self.train_sr_metric.update(x, batch['score'])
        self.batch_sr_metric.update(x, batch['score'])

        metric_output = self.batch_sr_metric.compute()
        self.batch_sr_metric.reset()
        self.log_dict({
            'Accuracy/Train/Step': metric_output['accuracy'],
            'Precision/Train/Step': metric_output['precision'],
            'Recall/Train/Step': metric_output['recall'],
            'F1/Train/Step': metric_output['f1'],
        })
        return loss

    def on_train_epoch_end(self) -> None:
        metric_output = self.train_sr_metric.compute()
        self.train_sr_metric.reset()
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
        batch_size = batch['input_ids'].size(0)
        x = self.encoder.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
        )
        loss = self.loss_fn(x, batch['score'])

        self.log('loss', loss, prog_bar=True, logger=False, sync_dist=True)
        self.log('Loss/Val', loss, sync_dist=True, batch_size=batch_size)

        self.val_sr_metric.update(x, batch['score'])
        return loss

    def on_validation_epoch_end(self) -> None:
        metric_output = self.val_sr_metric.compute()
        self.val_sr_metric.reset()
        self.log_dict(
            {
                'Accuracy/Val': metric_output['accuracy'],
                'Precision/Val': metric_output['precision'],
                'Recall/Val': metric_output['recall'],
                'F1/Val': metric_output['f1'],
            },
            sync_dist=True
        )
