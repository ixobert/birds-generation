import os
from argparse import Namespace
import torch
import hydra
from omegaconf import DictConfig
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger as Logger
import networks



class VQEngine(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.net = networks.VQVAE(**self.hparams.net)

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)
    
    def compute_loss(self, target, output):
        return 0.0

    def training_step(self, train_batch, batch_idx):
        pass

    def validation_step(self, val_batch, batch_idx):
        pass

    def validation_end(self, outputs):
        pass

@hydra.main(config_path="configs", config_name="train_vqvae")
def main(cfg: DictConfig) -> None:
    print(cfg)

    if cfg.get('debug', False):
        logger = None
    else:
        logger = Logger(project=cfg['project_name'], name=cfg['run_name'], tags=cfg['tags'])

    engine = VQEngine(Namespace(**cfg))
    trainer = pl.Trainer(
        logger=logger,
        gpus=cfg.get('gpus', 0),
    )



if __name__ == "__main__":
    main()