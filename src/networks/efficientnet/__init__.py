__version__ = "0.7.0"
import pytorch_lightning as pl
import torch
from .model import EfficientNet, VALID_MODELS
from .utils import (
    GlobalParams,
    BlockArgs,
    BlockDecoder,
    efficientnet,
    get_model_params,
)

class EfficientNetPl(pl.LightningModule):
  def __init__(self,backbone_network, num_classes, pretrained=False):
    super().__init__()
    self.efficient_net = EfficientNet.from_pretrained(backbone_network, num_classes=num_classes)


  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(),lr = 1e-3)
    return optimizer

  def forward(self,x):
    out = self.efficient_net(x)
    return out