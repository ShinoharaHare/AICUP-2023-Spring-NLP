from typing import Optional, Union

import fire
import lightning as L
import torch.backends.cuda
import torch.backends.cudnn
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.loggers import WandbLogger

from aicup.data import CrossEncoderSentenceRetrievalRetrievalDataModule
from aicup.models import CrossEncoderSentenceRetriever

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def main(
    name: str = 'cross-encoder-sentence-retriever',
    version: str = 'u1m3lfd3',
    batch_size: int = 16,
    accumulate_grad_batches: int = 1,
    num_workers: int = 4,
    precision: str = '16-mixed',
    gradient_clip_val: int = 1.0,
    max_epochs: int = 10,
    val_check_interval: Optional[Union[int, float]] = None,
    save_every_n_steps: Optional[int] = None,
    ckpt_path: Optional[str] = 'logs/aicup-2023-spring-nlp/u1m3lfd3/checkpoints/e8.ckpt'
):
    # L.seed_everything(42)

    model = CrossEncoderSentenceRetriever()
    datamodule = CrossEncoderSentenceRetrievalRetrievalDataModule(
        dataset_path='data/sentence_retrieval/cross_encoder',
        wiki_dataset_path='data/wiki',
        tokenizer=model.tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model.configure_sharded_model()

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
            ModelCheckpoint(
                monitor='epoch',
                mode='max',
                filename='e{epoch}',
                auto_insert_metric_name=False,
                save_on_train_epoch_end=True,
                save_top_k=-1,
            ),
        ],
        max_epochs=max_epochs,
        val_check_interval=val_check_interval,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        reload_dataloaders_every_n_epochs=1,
        # enable_checkpointing=False,
        # max_steps=1
    )

    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )

    trainer.save_checkpoint(model.checkpoint_dir + '/latest.ckpt')


if __name__ == '__main__':
    fire.Fire(main)
