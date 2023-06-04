from typing import Optional

import fire
import torch


def main(
    checkpoint_path: str,
    output_path: Optional[str] = None
):
    checkpoint = torch.load(checkpoint_path, 'cpu')
    output_path = output_path or checkpoint_path.replace('.ckpt', '.weights.ckpt')
    keys_to_drop = ['MixedPrecisionPlugin', 'callbacks', 'lr_schedulers', 'optimizer_states']
    for k in keys_to_drop:
        if k in checkpoint:
            checkpoint.pop(k)
    torch.save(checkpoint, output_path)

if __name__ == '__main__':
    fire.Fire(main)
