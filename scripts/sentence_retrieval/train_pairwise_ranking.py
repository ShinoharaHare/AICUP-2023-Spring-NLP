from typing import Optional, Union

import fire
import lightning as L
import torch.backends.cuda
import torch.backends.cudnn
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.loggers import WandbLogger

from aicup.data import PairwiseRankingRetrievalDataModule
from aicup.models import PairwiseRankingSentenceRetriever

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# torch.backends.cudnn.benchmark = True


def main(
    dataset_path: str,
    claim_dataset_path: str,
    wiki_dataset_path: str,
    name: Optional[str] = None,
    version: Optional[str] = None,
    batch_size: int = 16,
    accumulate_grad_batches: int = 1,
    num_workers: int = 4,
    precision: str = '16-mixed',
    gradient_clip_val: int = 1.0,
    max_epochs: int = 3,
    val_check_interval: Optional[Union[int, float]] = None,
    ckpt_path: Optional[str] = None,
):
    model = PairwiseRankingSentenceRetriever(
        model_path='IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese',
        claim_dataset_path=claim_dataset_path,
        wiki_dataset_path=wiki_dataset_path,
        _load_from_checkpoint=ckpt_path is not None,
    )

    model.save_hyperparameters(dict(
        dataset_path=dataset_path,
        batch_size=batch_size,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
        max_epochs=max_epochs,
    ))

    datamodule = PairwiseRankingRetrievalDataModule(
        dataset_path=dataset_path,
        tokenizer=model.tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    trainer = L.Trainer(
        precision=precision,
        logger=[
            WandbLogger(
                project='aicup-2023-spring-nlp',
                save_dir='logs',
                name=name,
                version=version,
            )
        ],
        callbacks=[
            LearningRateMonitor(),
            # EarlyStopping(
            #     monitor='Accuracy/Val',
            #     mode='max'
            # ),
            ModelCheckpoint(
                monitor='epoch',
                mode='max',
                filename='e{epoch}',
                auto_insert_metric_name=False,
                save_on_train_epoch_end=True,
                save_top_k=-1,
            ),
            ModelCheckpoint(
                monitor='Accuracy/Val',
                mode='max',
                filename='best-{Accuracy/Val:.4f}',
                auto_insert_metric_name=False,
                save_weights_only=True,
            )
        ],
        max_epochs=max_epochs,
        val_check_interval=val_check_interval,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        reload_dataloaders_every_n_epochs=1,
    )

    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )


if __name__ == '__main__':
    fire.Fire(main)
