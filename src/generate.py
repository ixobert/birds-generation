import argparse
import os
from collections import OrderedDict

import torch
from torchvision.utils import save_image
from tqdm import tqdm

try:
    from networks import VQVAE
    from networks import PixelSNAIL
    from train_prior.PriorEngine import _label_to_dense_tensor
except ImportError:
    from src.networks import VQVAE
    from src.networks import PixelSNAIL
    from src.train_prior.PriorEngine import _label_to_dense_tensor

def _update_model_keys(old_model:OrderedDict):
    new_model = OrderedDict()
    for key,value in old_model.items():
        if key.startswith('net.'):
            new_model[key.replace('net.','', 1)] = value
        else:
            new_model[key] = value
    return new_model


#TODO: Detach Row before return
@torch.no_grad()
def sample_model(model, device, batch, size, temperature, condition=None):
    row = torch.zeros(batch, *size, dtype=torch.int64).to(device)
    cache = {}

    for i in tqdm(range(size[0])):
        for j in range(size[1]):
            out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache)
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            sample = torch.multinomial(prob, 1).squeeze(-1)
            row[:, i, j] = sample
    return row


def load_model(model, checkpoint=None, device='cuda'):
    if checkpoint:
        ckpt = torch.load(os.path.join('checkpoint', checkpoint), map_location=device)
    else: 
        print("No model provided.")
        ckpt = {}

    
    hparams = {}
    if 'hyper_parameters' in ckpt:
        hparams = ckpt['hyper_parameters']

    if model == 'vqvae':
        model = VQVAE(in_channel=1)

    elif model == 'pixelsnail_top':
        model = PixelSNAIL(
            [16, 16], # [32, 32],
            hparams['net']['n_class'],
            hparams['net']['channel'],
            hparams['net']['kernel_size'],
            hparams['net']['n_block'],
            hparams['net']['n_res_block'],
            hparams['net']['res_channel'],
            attention=hparams['net']['attention'],
            dropout=hparams['net']['dropout'],
            n_cond_res_block=hparams['net']['n_cond_res_block'],
            cond_res_channel=hparams['net']['cond_res_channel'],
            cond_res_kernel=hparams['net']['cond_res_kernel'],
            n_out_res_block=hparams['net']['n_out_res_block'],
        )

    elif model == 'pixelsnail_bottom':
        model = PixelSNAIL(
            [32, 32], # [64, 64],
            hparams['net']['n_class'],
            hparams['net']['channel'],
            hparams['net']['kernel_size'],
            hparams['net']['n_block'],
            hparams['net']['n_res_block'],
            hparams['net']['res_channel'],
            attention=hparams['net']['attention'],
            dropout=hparams['net']['dropout'],
            n_cond_res_block=hparams['net']['n_cond_res_block'],
            cond_res_channel=hparams['net']['cond_res_channel'],
            cond_res_kernel=hparams['net']['cond_res_kernel'],
            n_out_res_block=hparams['net']['n_out_res_block'],
        )
        
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
        ckpt = _update_model_keys(ckpt)
    

    if checkpoint:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    return model


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--vqvae', type=str)
    parser.add_argument('--top', type=str)
    parser.add_argument('--bottom', type=str)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--filename', type=str, default='temp-audio.png')

    args = parser.parse_args()

    model_vqvae = load_model('vqvae', args.vqvae, device)
    model_top = load_model('pixelsnail_top', args.top, device)
    model_bottom = load_model('pixelsnail_bottom', args.bottom, device)

    top_sample = sample_model(model_top, device, args.batch, [16, 16], args.temp)
    bottom_sample = sample_model(
        model_bottom, device, args.batch, [32, 32], args.temp, condition=top_sample
    )

    decoded_sample = model_vqvae.decode_code(top_sample, bottom_sample)
    decoded_sample = decoded_sample.clamp(-1, 1)
    torch.save(decoded_sample, 'generated.out')

    # save_image(decoded_sample, args.filename, normalize=True, range=(-1, 1))
