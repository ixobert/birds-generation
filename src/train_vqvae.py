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
import wandb



class VQEngine(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.net = networks.VQVAE(**self.hparams.net)
        self.criterion = nn.MSELoss()
        self.sample = None

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
        self.sample = img
        out, latent_loss = self.net(img)
        latent_loss = latent_loss.mean()


        loss, logs = self._compute_loss(target=img, output=out, latent_loss=latent_loss)
        result = pl.TrainResult(minimize=loss)
        result.log('loss', loss)
        result.log_dict(logs)
        self.logger.experiment.log(logs)
        return result

    def _generate(self, input):
        out = self.net(input, logits_only=True)
        return out

    def _remove_dim(self, input):
        """ 
        Remove extra dimension (dimension with only one vector)
        """
        if input.shape[0] == 1:
            return input.squeeze(1)
        return input
    
    def _convert_grid_to_img(self, grid, name, save=False):
        from PIL import Image
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        if save:
            im.save(name)
        return ndarr

    def training_epoch_end(self,*args, **kwargs):
        out = self._generate(self.sample)
        out = self._remove_dim(out)

        os.makedirs('./outs', exist_ok=True)
        os.makedirs('./samples', exist_ok=True)
        torch.save(out, f'./outs/{str(self.current_epoch)}.tmp')
        torch.save(out, f'./samples/{str(self.current_epoch)}.tmp')

        input_grid = make_grid(self._remove_dim(self.sample).cpu(), nrow=self.sample.shape[0], padding=True, pad_value=1.0)
        recon_grid = make_grid(out.detach().cpu(), nrow=self.sample.shape[0], padding=True, pad_value=1.0)

        input_grid = self._convert_grid_to_img(input_grid, './input.png', save=True) #Need to set save to True if you want to save the image
        recon_grid = self._convert_grid_to_img(recon_grid, './recon.png', save=True)
        
        self.logger.experiment.log({
            'input':         wandb.Image(input_grid),
            'reconstructed': wandb.Image(recon_grid),
        }, self.current_epoch)

        return {}




@hydra.main(config_path="configs", config_name="train_vqvae")
def main(cfg: DictConfig) -> None:
    print(cfg)
    current_folder = os.getcwd()
    print("Current Folder", current_folder)

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
    checkpoint_callback = ModelCheckpoint('./models', monitor='loss', verbose=True)
    trainer = pl.Trainer(
        logger=logger,
        gpus=cfg.get('gpus', 0),
        max_epochs=cfg.get('nb_epochs', 3),
        checkpoint_callback=checkpoint_callback,
    )

    # Start training
    trainer.fit(engine, train_dataloader=train_dataloader)

if __name__ == "__main__":
    main()