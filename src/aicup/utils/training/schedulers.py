import math

from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer


__all__ = ['get_cosine_scheduler', 'get_warmup_scheduler']

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

def get_warmup_scheduler(
    optimizer: Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
):
    return LambdaLR(optimizer, lambda cs: min(cs / max(1, num_warmup_steps), 1.0), last_epoch)
