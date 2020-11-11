import logging
import os
from math import radians
from threading import Condition

os.environ['HYDRA_FULL_ERROR'] = '1'
from argparse import Namespace

import hydra
import lmdb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger as Logger
from torch import nn
from torchvision.utils import make_grid

import networks
from dataloaders import CodeBookDataModule


class PriorEngine(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.net = networks.PixelSNAIL(**self.hparams.net)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)
    
    def configure_optimizers(self,):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        return optimizer
    

    def _compute_loss(self, target, output):
        loss = self.criterion(output, target)
        logs = {
            'loss': loss,
        }
        return loss, logs


    def _filepath_to_label(self, filepath:str) -> int:
        """Extract label from filepath and return the label as an integer.

        Args:
            filepath ([str]): [description]

        Returns:
            int: [description]
        """
        class_name = filepath.split('/')[-2]
        label = self.hparams.dataset.classes_name.index(class_name)

        return label

    @classmethod
    def _label_to_dense_tensor(self, labels, shape):
        out = []
        shape = [_ for _ in shape]
        shape[0] = 1
        for label in labels:
            temp=torch.Tensor()
            torch.full(shape, label, out=temp)
            out.append(temp)
        out = torch.cat(out).to(torch.int64)
        return out

    def _step(self, batch, batch_idx):
        top, bottom, filepaths = batch
        label_idx = torch.tensor([self._filepath_to_label(x) for x in filepaths]) + 1
        label_idx = label_idx.to(top.device)

        if self.hparams.net.model_type == 'top':
            target = top
            label_tensor = self._label_to_dense_tensor(label_idx, torch.tensor(top.shape)//2)
            out, _ = self.net(top, condition=label_tensor, condition_label=label_idx)
        elif self.hparams.net.model_type == 'bottom':
            target = bottom 
            out, _ = self.net(bottom, condition=top, condition_label=label_idx)
        else:
            print("Only top and bottom are supported for model_type") 
            raise ValueError
        loss, logs = self._compute_loss(target=target, output=out)
        logs['accuracy'] =  self._get_accuracy(out, target)
        return out, target, loss, logs
      
 
    def _get_accuracy(self, out, target):
        _, pred = out.max(1)
        correct = (pred == target).float()
        accuracy = correct.sum() / target.numel()
        return accuracy


    def training_step(self, batch, batch_idx) :
        out, target, loss, logs = self._step(batch, batch_idx)
        self.log_dict(logs)
        self.logger.experiment.log(logs)
        return logs


    def training_epoch_end(self, outputs) -> dict:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['accuracy'] for x in outputs]).mean()
        self.log_dict({'loss': avg_loss, 'accuracy': avg_accuracy})

        


@hydra.main(config_path="configs", config_name="train_prior")
def main(cfg: DictConfig) -> None:
    cfg['net']['num_classes_labels'] = len(cfg['dataset']['classes_name'])
    if cfg['net']['model_type'] =='top':
        cfg['net']['shape'] = [16, 16]
    else:
        cfg['net']['shape'] = [32, 32]
    logging.info(cfg)
    current_folder = os.getcwd()
    print("Current Folder", current_folder)

    if cfg.get('debug', False):
        logger = Logger(project=cfg['project_name'], name=cfg['run_name'], tags=cfg['tags']+["debug"])
    else:
        logger = Logger(project=cfg['project_name'], name=cfg['run_name'], tags=cfg['tags'])

    train_dataloader = CodeBookDataModule(config=cfg['dataset'])

    engine = PriorEngine(Namespace(**cfg))
    checkpoint_callback = ModelCheckpoint(f"./models-prior-{cfg['net']['model_type']}", monitor='loss', verbose=True)
    trainer = pl.Trainer(
        logger=logger,
        gpus=cfg.get('gpus', 0),
        max_epochs=cfg.get('nb_epochs', 3),
        checkpoint_callback=checkpoint_callback,
    )
    
    logger.log_hyperparams(cfg)

    # Start training
    trainer.fit(engine, train_dataloader=train_dataloader)

if __name__ == "__main__":
    main()
