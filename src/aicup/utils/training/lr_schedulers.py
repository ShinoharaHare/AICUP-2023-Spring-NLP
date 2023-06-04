import inspect
import math
from typing import Literal, Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

__all__ = ['get_cosine_scheduler', 'get_linear_scheduler', 'get_warmup_scheduler', 'get_lr_scheduler']

def get_cosine_scheduler(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    min_lr_factor: float = 0.0
):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        factor = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        factor = (1.0 - min_lr_factor) * (factor - 0.0) + min_lr_factor
        return factor
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_linear_scheduler(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
    min_lr_factor: float = 0.0
):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        factor = max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
        factor = (1.0 - min_lr_factor) * (factor - 0.0) + min_lr_factor
        return factor
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_warmup_scheduler(
    optimizer: Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
):
    return LambdaLR(optimizer, lambda cs: min(cs / max(1, num_warmup_steps), 1.0), last_epoch)


def get_lr_scheduler(
    scheduler_type: Literal[None, 'linear', 'cosine'],
    optimizer: Optimizer,
    num_warmup_steps: int = 0,
    num_training_steps: Optional[int] = None,
    min_lr_factor: Optional[float] = None,
):
    mapping = {
        None: get_warmup_scheduler,
        'linear': get_linear_scheduler,
        'cosine': get_cosine_scheduler,
    }
    getter = mapping[scheduler_type]

    lr_scheduler_kwargs = dict(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_factor=min_lr_factor,
    )
    lr_scheduler_kwargs = {k: lr_scheduler_kwargs[k] for k in inspect.signature(getter).parameters if k in lr_scheduler_kwargs}
    return getter(**lr_scheduler_kwargs)
