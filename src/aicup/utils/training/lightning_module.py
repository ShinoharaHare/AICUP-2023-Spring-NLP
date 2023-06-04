
import os
from typing import IO, Any, Dict, List, Literal, Optional, Tuple, Union

import lightning as L
from lightning.fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor, nn, optim
from torch.nn.modules.module import _IncompatibleKeys
from typing_extensions import Self

from .lr_schedulers import get_lr_scheduler


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
    
    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[_PATH, IO],
        map_location: _MAP_LOCATION_TYPE = None,
        hparams_file: Optional[_PATH] = None,
        strict: bool = True,
        **kwargs: Any
    ) -> Self:
        return super().load_from_checkpoint(
            checkpoint_path,
            map_location,
            hparams_file,
            strict,
            load_from_checkpoint=True,
            **kwargs
        )
    
    def print_rank0(self, *args, **kwargs) -> None:
        if self.is_global_zero:
            print(*args, **kwargs)

    def __init__(
        self,
        learning_rate: float = 1e-5,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        optim_bits: Union[int, str] = 32,
        lr_scheduler_type: Literal[None, 'linear', 'cosine'] = None,
        num_warmup_steps: int = 0,
        min_lr_factor: float = 0.1,
        _load_from_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        self._register_state_dict_hook(_state_dict_hook)
        self.register_load_state_dict_post_hook(_load_state_dict_post_hook)

        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.optim_bits = optim_bits
        self.lr_scheduler_type = lr_scheduler_type
        self.num_warmup_steps = num_warmup_steps
        self.min_lr_factor = min_lr_factor

        self._load_from_checkpoint = _load_from_checkpoint

    def configure_optimizers(self):
        optimizer_config = {}
        parameters = [p for p in self.parameters() if p.requires_grad]
        
        optimizer_kwargs = dict(
            params=parameters,
            lr=self.learning_rate,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            optim_bits=self.optim_bits
        )

        if self.optim_bits == '32-torch':
            optimizer_cls = optim.AdamW
            optimizer_kwargs.pop('optim_bits')
        else:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW

        optimizer_config['optimizer'] = optimizer_cls(**optimizer_kwargs)
        optimizer_config['lr_scheduler'] = {
            'scheduler': get_lr_scheduler(
                scheduler_type=self.lr_scheduler_type,
                optimizer=optimizer_config['optimizer'],
                num_warmup_steps=self.num_warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
                min_lr_factor=self.min_lr_factor,
            ),
            'interval': 'step',
        }
        return optimizer_config
