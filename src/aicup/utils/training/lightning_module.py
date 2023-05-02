
import os
from typing import Dict, List

# import deepspeed
import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DeepSpeedStrategy
from torch import Tensor, nn
from torch.nn.modules.module import _IncompatibleKeys
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from ..safetensors import SafeOpen
from .pf_embedding import PartiallyFrozenEmbedding
from .pf_linear import PartiallyFrozenLinear


def _state_dict_hook(module: nn.Module, state_dict: Dict[str, Tensor], prefix: str, local_metadata):
    trainables = set(n for n, p in module.named_parameters() if p.requires_grad)
    for k in list(state_dict.keys()):
        if k not in trainables:
            state_dict.pop(k)

def _load_state_dict_post_hook(module: nn.Module, incompatible_keys: _IncompatibleKeys):
    missing_keys = set(incompatible_keys.missing_keys) & set(n for n, p in module.named_parameters() if p.requires_grad)
    incompatible_keys.missing_keys[:] = list(missing_keys)

class LightningModuleX(L.LightningModule):
    logger: WandbLogger

    @property
    def checkpoint_dir(self):
        return os.path.join(self.logger.save_dir, self.logger.name, self.logger.version, 'checkpoints')
    
    @property
    def is_global_zero(self):
        return self.trainer.is_global_zero
    
    @property
    def strategy(self):
        return self.trainer.strategy
    
    @property
    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    @property
    def trainable_parameter_names(self) -> List[str]:
        return [n for n, p in self.named_parameters() if p.requires_grad]
        
    def __init__(self) -> None:
        super().__init__()

        self._register_state_dict_hook(_state_dict_hook)
        self.register_load_state_dict_post_hook(_load_state_dict_post_hook)
    
    def print_rank0(self, *args, **kwargs) -> None:
        if self.is_global_zero:
            print(*args, **kwargs)

    # def load_safetensors(self, module: nn.Module, path: str):
    #     pbar = tqdm(total=len(list(module.parameters())), desc='Loading weights', disable=not self.is_global_zero)
    #     f = SafeOpen(path)
    #     for prefix, m in module.named_modules():
    #         params: Dict[str, Tensor] = {}
    #         for n, p in m.named_parameters(prefix=prefix, recurse=False):
    #             if not getattr(p, '__loaded', False):
    #                 params[n] = p
    #                 setattr(p, '__loaded', True)

    #         if not params:
    #             continue
            
    #         ds_zero3 = isinstance(self.strategy, DeepSpeedStrategy) and self.strategy.zero_stage_3   
    #         with deepspeed.zero.GatheredParameters(params.values(), modifier_rank=0, enabled=ds_zero3):
    #             if self.is_global_zero or not ds_zero3:
    #                 for n, p in params.items():
    #                     t = f.get_tensor(n)
    #                     p.data.copy_(t)
    #                     pbar.update()
    #     f.close()

    def resize_token_embeddings(
        self,
        model: PreTrainedModel,
        new_num_tokens: int,
        freeze_old: bool = False,
        initialize_from_old_tokens: bool = False
    ):
        old_num_tokens = model.config.vocab_size
        model.resize_token_embeddings(new_num_tokens)
        input_embeddings: nn.Embedding = model.get_input_embeddings()
        output_embeddings: nn.Linear = model.get_output_embeddings()
        
        if initialize_from_old_tokens:
            mask = torch.randperm(old_num_tokens, device=model.device)[:new_num_tokens - old_num_tokens]
            input_embeddings.weight.data[old_num_tokens:].copy_(input_embeddings.weight[mask])
            output_embeddings.weight.data[old_num_tokens:].copy_(output_embeddings.weight[mask])

        if freeze_old:
            model.set_input_embeddings(PartiallyFrozenEmbedding(input_embeddings, old_num_tokens))
            model.set_output_embeddings(PartiallyFrozenLinear(output_embeddings, old_num_tokens))
