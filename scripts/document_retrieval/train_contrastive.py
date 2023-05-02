from typing import Optional, Union

import fire
import lightning as L
import torch.backends.cuda
import torch.backends.cudnn
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from aicup.data import ContrastiveDocumentRetrieverDataModule
from aicup.models import ContrastiveDocumentRetriever

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def main(
    name: str = 'contrastive-document-retriever',
    version: str = '8zg0qvu1',
    batch_size: int = 16,
    accumulate_grad_batches: int = 1,
    num_workers: int = 4,
    precision: str = '16-mixed',
    gradient_clip_val: int = 1.0,
    max_epochs: int = 10,
    val_check_interval: Optional[Union[int, float]] = None,
    save_every_n_steps: Optional[int] = None,
    ckpt_path: Optional[str] = 'logs/aicup-2023-spring-nlp/8zg0qvu1/checkpoints/e0.ckpt',
):
    model = ContrastiveDocumentRetriever()
    model.configure_sharded_model()

    datamodule = ContrastiveDocumentRetrieverDataModule(
        dataset_path='data/document_retrieval/contrastive',
        wiki_dataset_path='data/wiki',
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
    )

    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )

    trainer.save_checkpoint(model.checkpoint_dir + '/latest.ckpt')


if __name__ == '__main__':
    fire.Fire(main)
