import logging
import os
os.environ['HYDRA_FULL_ERROR'] = '1'
from argparse import Namespace
import torch
from torchvision.utils import make_grid
import hydra
from omegaconf import DictConfig
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger as Logger
from pytorch_lightning.callbacks import ModelCheckpoint
import networks
from dataloaders import SpectrogramsDataModule
from dataloaders import ImagesDataModule
import wandb
import lmdb
from utils.helpers import extract_latent



class VQEngine(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.net = networks.VQVAE(**self.hparams.net)
        self.criterion = nn.MSELoss()
        self.cache = {}

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)
    
    def configure_optimizers(self,):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        return optimizer
    

    def _compute_loss(self, target, output, latent_loss):
        recon_loss = self.criterion(output, target)
        loss = recon_loss + self.hparams['latent_loss_weight'] * latent_loss
        logs = {
            'mse': recon_loss,
            'latent_loss': latent_loss,
            'loss': loss,
        }
        return loss, logs


    def training_step(self, train_batch, batch_idx):
        img, label, file_path = train_batch
        if 'sample' not in self.cache:
            self.cache['sample'] = img

        out, latent_loss = self.net(img)
        latent_loss = latent_loss.mean()

        loss, logs = self._compute_loss(target=img, output=out, latent_loss=latent_loss)
        self.log('loss', loss)
        self.log_dict(logs)
        self.logger.experiment.log(logs)
        return logs

    def _generate(self, input):
        quant_top, quant_bottom, diff, id_top, id_bottom = self.net.encode(input)
        out = self.net.decode(quant_top, quant_bottom)

        return out.detach().cpu(), id_top.long().detach().cpu(), id_bottom.long().detach().cpu()

    def _remove_dim(self, input):
        """ 
        Remove extra dimension (dimension with only one vector)
        """
        if input.shape[0] == 1:
            return input.squeeze(1)
        return input
    
    def _convert_grid_to_img(self, grid, outfile=None):
        ndarr = grid.permute(1,2,0).numpy()
        if outfile:
            from PIL import Image
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu').numpy()
            im = ndarr.astype('uint8')
            im = Image.fromarray(im)
            im.save(outfile)
        return ndarr

    def _render_codebook(self, codebook):
        import matplotlib.pyplot as plt
        import io
        fig, axs = plt.subplots()
        axs.imshow(codebook[0], interpolation=None)
        for i in range(len(codebook[0][0])):
            for j in range(len(codebook[0][1])):
                axs.text(i,j, codebook[0][j][i],ha='center',va='center')
        return fig


    def training_epoch_end(self,*args, **kwargs):
        # epoch = self.current_epoch
        if self.current_epoch % 2 != 0:
            return 
        if 'sample' not in self.cache:
            return
        sample = self.cache['sample']

        out, codebook_top, codebook_bottom = self._generate(sample)

        # os.makedirs('./outs', exist_ok=True)
        # os.makedirs('./samples', exist_ok=True)
        # torch.save(out, f'./outs/{str(self.current_epoch)}.tmp')
        # torch.save(sample, f'./samples/{str(self.current_epoch)}.tmp')
        input_grid = make_grid(self._remove_dim(sample), nrow=len(sample), padding=True, pad_value=1.0)
        recon_grid = make_grid(self._remove_dim(out), nrow=len(sample), padding=True, pad_value=1.0)
        

        input_grid = self._convert_grid_to_img(input_grid) #Need to set save to True if you want to save the image
        recon_grid = self._convert_grid_to_img(recon_grid)

        self.logger.experiment.log({
            'spectrograms':[
                wandb.Image(input_grid, caption='Input'), 
                wandb.Image(recon_grid, caption='Reconstructed'),
                ]
        })



@hydra.main(config_path="configs", config_name="train_vqvae")
def main(cfg: DictConfig) -> None:

    # _git_hash_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'git_hash.txt')
    # print(_git_hash_file)
    # with open(_git_hash_file) as reader:
    #     _git_hash = reader.read().strip()
    #     print(f"Git hash ***{_git_hash}***")
    #     cfg['git_hash'] = _git_hash

    if cfg.get('debug', False):
        logger = Logger(project=cfg['project_name'], name=cfg['run_name'], tags=cfg['tags']+["debug"])
    else:
        logger = Logger(project=cfg['project_name'], name=cfg['run_name'], tags=cfg['tags'])


    train_dataloader = SpectrogramsDataModule(config=cfg['dataset'])

    engine = VQEngine(Namespace(**cfg))
    if cfg.get('pretrained_weights', ''):
        engine.load_from_checkpoint(checkpoint_path=cfg['pretrained_weights'])
    checkpoint_callback = ModelCheckpoint('./models-vqvae', monitor='loss', verbose=True)
    trainer = pl.Trainer(
        logger=logger,
        gpus=cfg.get('gpus', 0),
        max_epochs=cfg.get('nb_epochs', 3),
        checkpoint_callback=checkpoint_callback,
    )
    logger.log_hyperparams(cfg)

    logging.info(cfg)
    current_folder = os.getcwd()
    logging.info(f"Current Folder:{current_folder}")

    if 'train' in cfg.get('mode'):
        # Start training
        trainer.fit(engine, train_dataloader=train_dataloader)

    if 'extract' in cfg.get('mode'):
        logging.info("Extract Latent Codes")
        train_dataloader.setup()
        # Extract latent variables from the training samples.
        map_size = 1000 * 1024*1024*1024
        env = lmdb.open('./latents.lmdb', map_size=map_size)
        extract_latent(lmdb_env=env, net=engine.net, dataloader=train_dataloader.train_dataloader())
        

if __name__ == "__main__":
    main()