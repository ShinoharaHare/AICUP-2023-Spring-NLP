import re
from typing import List, Literal, Optional, Tuple, TypedDict, Union, cast

import torch
from opencc import OpenCC
from torch import FloatTensor, LongTensor, nn, optim
from torchmetrics import Accuracy
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, DebertaV2PreTrainedModel,
                          PretrainedConfig, PreTrainedModel,
                          PreTrainedTokenizerFast)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import no_init_weights

from ...utils.training import LightningModuleX, get_lr_scheduler


class _BatchType(TypedDict):
    input_ids: LongTensor
    attention_mask: LongTensor
    token_type_ids: LongTensor
    labels: FloatTensor


def resize_token_embeddings(model: PreTrainedModel, n: int):
    model.resize_token_embeddings(n)
    return model


def resize_position_embeddings(model: PreTrainedModel, n: int):
    old_position_embeddings: nn.Embedding = model.base_model.embeddings.position_embeddings
    new_embedding_weight = old_position_embeddings.weight.data.mean(0)
    pivot = model.config.max_position_embeddings
    model.config.max_position_embeddings = n
    position_embeddings = nn.Embedding(model.config.max_position_embeddings, model.config.hidden_size)
    model.base_model.base_model.embeddings.position_ids = torch.arange(model.config.max_position_embeddings).expand((1, -1))
    position_embeddings.weight.data[:pivot].copy_(model.base_model.base_model.embeddings.position_embeddings.weight.data)
    position_embeddings.weight.data[pivot:].copy_(new_embedding_weight)
    model.base_model.base_model.embeddings.position_embeddings = position_embeddings
    return model


class ClassifierClaimVerifier(LightningModuleX):

    @property
    def config(self) -> PretrainedConfig:
        return self.base_model.config

    def __init__(
        self,
        model_path: str = 'IDEA-CCNL/Erlangshen-MegatronBert-1.3B-NLI',
        max_position_embeddings: int = 512,
        learning_rate: float = 3e-5,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        optim_bits: int = 32,
        lr_scheduler_type: Literal[None, 'linear', 'cosine'] = 'cosine',
        num_warmup_steps: int = 300,
        min_lr_factor: float = 0.1,
        _load_from_checkpoint: bool = False,
    ) -> None:
        super().__init__(learning_rate, betas, eps, weight_decay, optim_bits, lr_scheduler_type, num_warmup_steps, min_lr_factor, _load_from_checkpoint)

        self.save_hyperparameters(ignore=['_load_from_checkpoint'])

        self.model_path = model_path
        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        self.tokenizer.model_max_length = max_position_embeddings
        self.tokenizer.add_tokens('<TITLE>', special_tokens=True)
        
        if self._load_from_checkpoint:
            config = AutoConfig.from_pretrained(self.model_path, num_labels=3)
            with no_init_weights():
                self.base_model = AutoModelForSequenceClassification.from_config(config).cuda()
        else:
            self.base_model = AutoModelForSequenceClassification.from_pretrained(self.model_path, num_labels=3, low_cpu_mem_usage=True)

        self.base_model = cast(PreTrainedModel, self.base_model).cuda()
        self.base_model = resize_token_embeddings(self.base_model, len(self.tokenizer))
        self.base_model.classifier.reset_parameters()
        
        self.base_model.gradient_checkpointing_enable()

        self.train_accuracy = Accuracy(task='multiclass', num_classes=3)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=3)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def training_step(self, batch: _BatchType, batch_idx: int):
        batch_size = batch['input_ids'].size(0)
        
        outputs: SequenceClassifierOutput = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids'],
            labels=batch['labels'],
        )

        loss = outputs.loss

        self.log('global_step', self.global_step + 1, prog_bar=True, logger=False)
        self.log('loss', loss, prog_bar=True, logger=False)
        self.log('Loss/Train', loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('Loss/Train/Step', loss)
        
        self.train_accuracy.update(outputs.logits, batch['labels'])
        accuracy = self.train_accuracy.compute()
        self.log('Accuracy/Train/Step', accuracy)
        self.log('Accuracy/Train', accuracy, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.train_accuracy.reset()
        return loss

    def validation_step(self, batch: _BatchType, batch_idx: int):
        batch_size = batch['input_ids'].size(0)
        outputs: SequenceClassifierOutput = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids'],
            labels=batch['labels'],
        )

        loss = outputs.loss

        self.log('loss', loss, prog_bar=True, logger=False, sync_dist=True, batch_size=batch_size)
        self.log('Loss/Val', loss, sync_dist=True, batch_size=batch_size)

        self.val_accuracy.update(outputs.logits, batch['labels'])
        self.log('Accuracy/Val', self.val_accuracy)
        return loss
    
    @torch.inference_mode()
    def predict(
        self,
        batch_claim: List[str],
        batch_titles: List[List[str]],
        batch_sentences: List[List[str]],
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        if not hasattr(self, 't2s'):
            self.t2s = OpenCC('t2s')
        batch_claim_chs = [self.t2s.convert(c) for c in batch_claim]
        batch_evidences_chs = []
        for titles, sentences in zip(batch_titles, batch_sentences):
            for i in range(len(titles)):
                sentence = re.sub(r'(\t.*)+$', '', sentences[i])
                sentences[i] = f'{titles[i]} <TITLE> {sentence}'
            evidence = f' {self.tokenizer.sep_token} '.join(sentences)
            batch_evidences_chs.append(self.t2s.convert(evidence))

        batch_encoding = self.tokenizer(
            batch_claim_chs,
            batch_evidences_chs,
            return_tensors='pt',
            padding=True,
            max_length=max_length,
            truncation=max_length is not None,
        )
        batch_encoding = batch_encoding.to(self.device)
        outputs: SequenceClassifierOutput = self(**batch_encoding)
        preds = outputs.logits.argmax(-1)
        return preds
