from typing import List, Literal, Optional, Tuple, TypedDict, Union

import torch
from opencc import OpenCC
from torch import BoolTensor, FloatTensor, LongTensor, Tensor, nn
from transformers import BertTokenizerFast
from transformers.modeling_utils import no_init_weights

from ...utils.training import LightningModuleX
from ..ro_longformer.modeling_ro_longformer import (
    LongformerBaseModelOutputWithPooling, LongformerConfig, LongformerModel,
    LongformerTokenClassifierOutput)


class _BatchType(TypedDict):
    input_ids: LongTensor
    attention_mask: LongTensor
    token_type_ids: LongTensor
    labels: FloatTensor
    claim_mask: BoolTensor
    sentence_slices: List[List[slice]]
    is_evidence: List[List[bool]]


class TokenLevelSentenceRetriever(LightningModuleX):

    @property
    def config(self) -> LongformerConfig:
        return self.longformer.config
    
    def __init__(
        self,
        model_path: str = 'IDEA-CCNL/Erlangshen-Longformer-330M',
        pos_weight: Optional[float] = None,
        learning_rate: float = 1e-5,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        optim_bits: int = 32,
        lr_scheduler_type: Literal[None, 'linear', 'cosine'] = None,
        num_warmup_steps: int = 500,
        min_lr_factor: float = 0.1,
        _load_from_checkpoint: bool = False,
    ) -> None:
        super().__init__(learning_rate, betas, eps, weight_decay, optim_bits, lr_scheduler_type, num_warmup_steps, min_lr_factor, _load_from_checkpoint)

        self.save_hyperparameters()

        self.model_path = model_path
        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(self.model_path)
        
        add_pooling_layer = False
        if self._load_from_checkpoint:
            config = LongformerConfig.from_pretrained(self.model_path)
            with no_init_weights():
                self.longformer = LongformerModel(config, add_pooling_layer=add_pooling_layer)
        else:
            self.longformer: LongformerModel = LongformerModel.from_pretrained(self.model_path, add_pooling_layer=add_pooling_layer)

        self.longformer.gradient_checkpointing_enable()

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.to_score = nn.Linear(self.config.hidden_size, 1)

        pos_weight = torch.tensor(pos_weight) if pos_weight else pos_weight
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        claim_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, LongformerTokenClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if global_attention_mask is None:
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1

        outputs: LongformerBaseModelOutputWithPooling = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits: Tensor = self.to_score(sequence_output)
        logits = logits.squeeze(-1)

        loss = None
        if labels is not None:
            loss: Tensor = self.loss_fn(logits.flatten(), labels.flatten())
            valid_mask = ~claim_mask & attention_mask.bool()
            loss = loss.masked_fill(~valid_mask.flatten(), 0.0)
            loss = loss.sum() / valid_mask.sum()

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return LongformerTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )

    def training_step(self, batch: _BatchType, batch_idx: int):
        batch_size = batch['input_ids'].size(0)
        
        outputs: LongformerTokenClassifierOutput = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids'],
            labels=batch['labels'],
            claim_mask=batch['claim_mask'],
        )

        loss = outputs.loss
                
        self.log('global_step', self.global_step + 1, prog_bar=True, logger=False)
        self.log('loss', loss, prog_bar=True, logger=False)
        self.log('Loss/Train', loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('Loss/Train/Step', loss)
        return loss

    def on_validation_epoch_start(self) -> None:
        if not self.trainer.sanity_checking:
            self.half()
            self.log('Accuracy/Val', self.compute_accuracy())
            self.float()
    
    def validation_step(self, batch: _BatchType, batch_idx: int):
        return None

    @torch.inference_mode()
    def compute_metrics(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        sentence_slices: List[List[slice]],
        is_evidence: List[List[bool]],
        threshold: float = 0.5
    ):
        outputs: LongformerTokenClassifierOutput = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return self.compute_metrics_by_logits(outputs.logits, sentence_slices, is_evidence, threshold)

    @torch.inference_mode()
    def compute_metrics_by_logits(
        self,
        logits: torch.FloatTensor,
        batch_sentence_slices: List[List[slice]],
        batch_is_evidence: List[List[bool]],
        threshold: float = 0.5
    ):
        batch_tp = 0
        batch_fp = 0
        batch_fn = 0

        batch_scores = logits.sigmoid()
        for scores, sentence_slices, is_evidence in zip(batch_scores, batch_sentence_slices, batch_is_evidence):
            evidence_indices = set(i for i, b in enumerate(is_evidence) if b)
            predicted_indices = set()
            # cls_score = scores[0]
            for i, sentence_slice in enumerate(sentence_slices):
                sentence_score = scores[sentence_slice].mean()
                score = sentence_score
                # score = (0.65 * cls_score + 0.35 * sentence_score) / 2
                if score >= threshold:
                    predicted_indices.add(i)

            batch_tp += len(evidence_indices & predicted_indices)
            batch_fp += len(predicted_indices - evidence_indices)
            batch_fn += len(evidence_indices - predicted_indices)
        
        return {
            'tp': batch_tp,
            'fp': batch_fp,
            'fn': batch_fn,
        }

    @torch.inference_mode()
    def predict(self, batch_claim: List[str], batch_sentences: List[List[str]]) -> list[torch.Tensor]:
        t2s = OpenCC('t2s')

        batch_claim_chs = []
        batch_sentences_chs = []
        batch_sentences_slices = []
        for claim, sentences in zip(batch_claim, batch_sentences):
            sentence_slices = []
            document = ''
            s_ti = 0
            for s in sentences:
                document += t2s.convert(s) + '\n'
                e_ti = s_ti + len(self.tokenizer.tokenize(s))
                sentence_slices.append([s_ti, e_ti])
                s_ti = e_ti
            batch_claim_chs.append(t2s.convert(claim))
            batch_sentences_chs.append(document)
            batch_sentences_slices.append(sentence_slices)

        encoding = self.tokenizer(
            batch_claim_chs,
            batch_sentences_chs,
            return_tensors='pt',
            truncation=True,
            max_length=4096,
            padding=True
        )
        encoding = encoding.to(self.device)
        
        outputs: LongformerTokenClassifierOutput = self(**encoding)
        batch_scores = outputs.logits.sigmoid()

        batch_output_scores = []
        sequence_ids = encoding.sequence_ids()
        for bi in range(len(batch_scores)):
            sentences_slices = batch_sentences_slices[bi]
            try:
                offset = sequence_ids.index(1)
            except ValueError:
                output_scores = torch.zeros(len(batch_sentences[bi]))
                batch_output_scores.append(output_scores)
                continue

            output_scores = []
            for s_ti, e_ti in sentences_slices:
                ss = slice(s_ti + offset, e_ti + offset)
                sentence_score = batch_scores[bi, ss].mean().item()
                score = sentence_score
                output_scores.append(score)
            output_scores = torch.tensor(output_scores).nan_to_num(nan=0.0)
            batch_output_scores.append(output_scores)
        return batch_output_scores

    def compute_accuracy(self) -> float:
        import gc

        from datasets import disable_caching, load_from_disk

        from aicup.data import WikiDataset

        disable_caching()
        
        if not hasattr(self, 'ground_truth_mapping'):
            wiki_dataset = WikiDataset.load('data/wiki')
            wiki_dataset.load_elasticsearch_index()
            claim_dataset = load_from_disk('data/claim')

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
                    'sentences': [],
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

                    for document in batch_retrieved_documents[i]['lines']:
                        new_batch['claim'].append(batch['claim'][i])
                        new_batch['sentences'].append(document)
                return new_batch
            
            self.dr_results = claim_dataset['test'].map(mapper, remove_columns=claim_dataset['test'].column_names, batched=True)

            del wiki_dataset
            del claim_dataset

            gc.collect()
            torch.cuda.empty_cache()

        def mapper(batch: dict[str, list]):
            new_batch = {}
            batch_scores = self.predict(batch['claim'], batch['sentences'])
            new_batch['scores'] = batch_scores
            return new_batch
        
        sr_results = self.dr_results.map(mapper, batched=True, batch_size=64)

        mapping: dict[str, list] = {}
        for x in sr_results:
            l = mapping.setdefault(x['claim'], [])
            for sentence, score in zip(x['sentences'], x['scores']):
                l.append((sentence, score))

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
