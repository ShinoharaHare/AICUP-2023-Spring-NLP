import gc
from typing import Dict, List, Literal, Optional, Tuple, TypedDict, Union

import torch
from datasets import disable_caching, load_from_disk
from opencc import OpenCC
from torch import FloatTensor, nn
from transformers import BatchEncoding, BertTokenizerFast
from transformers.modeling_outputs import (BaseModelOutput,
                                           SequenceClassifierOutput)
from transformers.modeling_utils import no_init_weights
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    ContextPooler, DebertaV2Config, DebertaV2Model, StableDropout)

from ...data import WikiDataset
from ...utils.training import LightningModuleX


class _BatchType(TypedDict):
    positive: BatchEncoding
    negative: BatchEncoding
    labels: FloatTensor


class PairwiseRankingSentenceRetriever(LightningModuleX):

    @property
    def config(self) -> DebertaV2Config:
        return self.deberta.config
    
    def __init__(
        self,
        model_path: str,
        claim_dataset_path: Optional[str] = None,
        wiki_dataset_path: Optional[str] = None,
        margin: float = 1.0,
        learning_rate: float = 1e-5,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        optim_bits: int = 32,
        lr_scheduler_type: Literal[None, 'linear', 'cosine'] = 'linear',
        num_warmup_steps: int = 2000,
        min_lr_factor: float = 0.1,
        _load_from_checkpoint: bool = False
    ) -> None:
        super().__init__(learning_rate, betas, eps, weight_decay, optim_bits, lr_scheduler_type, num_warmup_steps, min_lr_factor, _load_from_checkpoint)

        self.save_hyperparameters(ignore=['_load_from_checkpoint'])

        self.model_path = model_path
        self.claim_dataset_path = claim_dataset_path
        self.wiki_dataset_path = wiki_dataset_path

        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(self.model_path)
        self.tokenizer.add_tokens('<TITLE>', special_tokens=True)
        
        if self._load_from_checkpoint:
            config = DebertaV2Config.from_pretrained(self.model_path)
            with no_init_weights():
                self.deberta = DebertaV2Model(config)
        else:
            self.deberta: DebertaV2Model = DebertaV2Model.from_pretrained(self.model_path)
        
        self.deberta.resize_token_embeddings(len(self.tokenizer))

        self.deberta.gradient_checkpointing_enable()

        self.pooler = ContextPooler(self.config)
        self.dropout = StableDropout(self.config.hidden_dropout_prob)
        self.scorer = nn.Linear(self.config.hidden_size, 1)

        self.loss_fn = nn.MarginRankingLoss(margin=margin)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs: BaseModelOutput = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = self.pooler(outputs.last_hidden_state)
        pooled_output = self.dropout(pooled_output)
        logits = self.scorer(pooled_output).squeeze(-1)

        loss = None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def training_step(self, batch: _BatchType, batch_idx: int):
        outputs1: SequenceClassifierOutput = self(**batch['positive'])
        outputs2: SequenceClassifierOutput = self(**batch['negative'])
        loss = self.loss_fn(outputs1.logits, outputs2.logits, batch['labels'])
                
        self.log('global_step', self.global_step + 1, prog_bar=True, logger=False)
        self.log('loss', loss, prog_bar=True, logger=False)
        self.log('Loss/Train', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('Loss/Train/Step', loss)
        return loss

    # def validation_step(self, batch: _BatchType, batch_idx: int):
    #     outputs1: SequenceClassifierOutput = self(**batch['positive'])
    #     outputs2: SequenceClassifierOutput = self(**batch['negative'])
    #     loss = self.loss_fn(outputs1.logits, outputs2.logits, batch['labels'])
                
    #     self.log('loss', loss, prog_bar=True, logger=False)
    #     self.log('Loss/Val', loss, sync_dist=True)

    #     self.val_precision.update(outputs1.logits, torch.ones_like(batch['labels']))
    #     self.val_precision.update(outputs2.logits, torch.zeros_like(batch['labels']))

    #     self.val_recall.update(outputs1.logits, torch.ones_like(batch['labels']))
    #     self.val_recall.update(outputs2.logits, torch.zeros_like(batch['labels']))

    #     self.val_f1.update(outputs1.logits, torch.ones_like(batch['labels']))
    #     self.val_f1.update(outputs2.logits, torch.zeros_like(batch['labels']))
    #     self.log('Precision/Val', self.val_precision)
    #     self.log('Recall/Val', self.val_recall)
    #     self.log('F1/Val', self.val_f1)
    #     return loss

    def on_validation_epoch_start(self) -> None:
        if not self.trainer.sanity_checking:
            self.half()
            self.log('Accuracy/Val', self.compute_accuracy())
            self.float()

    def validation_step(self, batch: _BatchType, batch_idx: int):
        return None
    
    @torch.inference_mode()
    def predict(self, batch_claim: List[str], batch_title: List[str], batch_sentence: List[str], max_length: Optional[int] = None) -> torch.Tensor:
        if not hasattr(self, 't2s'):
            self.t2s = OpenCC('t2s')

        encoding = self.tokenizer(
            [self.t2s.convert(x) for x in batch_claim],
            [self.t2s.convert(f'{title} <TITLE> {sentence}') for title, sentence in zip(batch_title, batch_sentence)],
            return_tensors='pt',
            padding=True,
            max_length=max_length,
            truncation=max_length is not None,
        )
        encoding = encoding.to(self.device)
        outputs: SequenceClassifierOutput = self(**encoding)
        scores = outputs.logits.sigmoid()
        return scores    

    def compute_accuracy(self) -> float:
        disable_caching()

        if not hasattr(self, 'ground_truth_mapping'):
            wiki_dataset = WikiDataset.load(self.wiki_dataset_path)
            wiki_dataset.load_elasticsearch_index()
            claim_dataset = load_from_disk(self.claim_dataset_path)

            self.ground_truth_mapping = {}
            for x in claim_dataset['test']:
                x['evidence'] = x['evidence'] or []
                sentences_group = []
                for eg in x['evidence']:
                    sentences = set()
                    for dt, li in zip(eg['document'], eg['line']):
                        sentences.add(wiki_dataset[dt]['lines'][li])
                    sentences_group.append(sentences)
                self.ground_truth_mapping[x['claim']] = sentences_group

            def mapper(batch: dict[str, list]):
                new_batch = {
                    'claim': [],
                    'title': [],
                    'sentence': [],
                }

                batch_retrieved_documents = wiki_dataset.retrieve(
                    batch['claim'],
                    top_k=3,
                    min_score=10,
                    return_by_noun=True,
                    merge_adjacent=True,
                    return_unmerged=True,
                )

                for i in range(len(batch['id'])):
                    if not batch_retrieved_documents[i]['id']:
                        continue

                    for title, sentences in zip(batch_retrieved_documents[i]['id'], batch_retrieved_documents[i]['lines']):
                        for sentence in sentences:
                            if not sentence:
                                continue

                            new_batch['claim'].append(batch['claim'][i])
                            new_batch['title'].append(title)
                            new_batch['sentence'].append(sentence)
                return new_batch
            
            self.dr_results = claim_dataset['test'].map(mapper, remove_columns=claim_dataset['test'].column_names, batched=True)

            del wiki_dataset
            del claim_dataset

            gc.collect()
            torch.cuda.empty_cache()

        def mapper(batch: Dict[str, list]):
            new_batch = {}
            batch_score = self.predict(batch['claim'], batch['title'], batch['sentence'])
            new_batch['score'] = batch_score
            return new_batch
        
        sr_results = self.dr_results.map(mapper, batched=True, batch_size=64)

        mapping: Dict[str, list] = {}
        for x in sr_results:
            l = mapping.setdefault(x['claim'], [])
            l.append((x['sentence'], x['score']))

        correct = 0
        total = 0
        for k, v in mapping.items():
            v.sort(key=lambda x: x[1], reverse=True)
            preds = set(x[0] for x in v[:5])
        
            if self.ground_truth_mapping[k]:
                for gt in self.ground_truth_mapping[k]:
                    if len(gt - preds) == 0:
                        correct += 1
                        break
            else:
                correct += 1
                
            total += 1

        return correct / total
