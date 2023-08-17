from collections import OrderedDict
from copy import deepcopy
import os
import random
import pytorch_lightning as pl
import logging
import os
from scipy import rand
from PIL import Image
from torch._C import device
os.environ['HYDRA_FULL_ERROR'] = '1'
from argparse import Namespace
import torch
import hydra
from omegaconf import DictConfig
import argparse
import glob
import librosa
try:
    from networks.vqvae2 import VQVAE
    from dataloaders import SpectrogramsDataModule
    from dataloaders.audiodataset import AudioDataset
except ImportError:
    from src.networks.vqvae2 import VQVAE
    from src.dataloaders import SpectrogramsDataModule
    from src.dataloaders.audiodataset import AudioDataset

import tqdm
import numpy as np


parser = argparse.ArgumentParser(description="Data Augmentator")
parser.add_argument('--data_paths', type=str, default="", help="Audio paths list. (*.png, *.npy, *.wav)")
parser.add_argument("--out_folder", type=str, help="Output folder for generated samples.")
parser.add_argument('--augmentations', default="noise")
parser.add_argument('--num_samples', type=int, default= 3)
parser.add_argument('--device', default='cpu')
parser.add_argument('--model_path', type=str, default="epoch=5-step=30005.ckpt")


class Augmentations():
    ecogen_augs = [
        'noise',
        'interpolation',
    ]

    specbase_augs = [
        'input_dropout',
        'input_noise',
        'input_stretching',
        'specaug', #specaug,
        'mixup',
        ]

    def __init__(self,):
        pass

    def encode(self, model, img, device='cuda:0'):
        img = img.to(device)
        quant_top, quant_bottom, diff, id_top, id_bottom = model.encode(img)
        return (quant_top,
            quant_bottom,
            id_top,
            id_bottom)

    def load_audio(self, audio_path, sr=16384, seconds=4):
        audio, _sr = librosa.load(audio_path)
        audio = librosa.resample(audio, _sr, sr)
        audio = librosa.util.fix_length(audio, seconds)
        return audio

    def load_sample_spectrogram(self, audio_path, window_length=16384*4, sr=16384, n_fft=1024):
        audio = self.load_audio(audio_path, sr, window_length)
        features = librosa.feature.melspectrogram(y=audio, n_fft=n_fft)
        features = librosa.power_to_db(features)

        if features.shape[0] % 2 != 0: 
            features = features[1:, :]
        if features.shape[1] % 2 != 0:
            features = features[:, 1:]
        return features


    def load_sample(self, filepath):
        print(filepath)
        if filepath.endswith('.npy'):
            spectrogram = np.load(filepath)
            spectrogram = np.pad(spectrogram,[(0,0), (0,4)], mode='edge')
        elif filepath.endswith('.png'):
            spectrogram = np.array(Image.open(filepath))
        elif filepath.endswith('.wav'):
            spectrogram = self.load_sample_spectrogram(filepath)
        else:
            raise NotImplementedError(f"Filetype {filepath} not supported.")

        spectrogram = np.expand_dims(spectrogram, 0)
        spectrogram = np.expand_dims(spectrogram, 0)

        print("Size:", spectrogram.shape)
        return spectrogram


    def decode(self, model, quant_top, quant_bottom):
        return model.decode(quant_top, quant_bottom).detach()


    def specbase_augment(self, all_samples_paths, ratio=0.1, out_folder="", generation_count=5, transform='input_dropout', device='cpu'):

        for i, file_path in tqdm.tqdm(enumerate(all_samples_paths)):
            waveform, sr = AudioDataset._get_sample(path=file_path)
            spectrogram_op = AudioDataset._get_spectrogram_operation(n_fft=1024, win_length=1024, hop_length=256, center=True, pad_mode="reflect", power=2.0)

            spectrogram = spectrogram_op(waveform)
            temp_spec = deepcopy(spectrogram)
            for j in range(generation_count):
                if  transform ==' input_noise':
                    temp_waveform = deepcopy(waveform) + ratio*torch.randn_like(waveform)
                    temp_spec = spectrogram_op(temp_waveform)
                    reconstructed = SpectrogramsDataModule.custom_augment_torchaudio(temp_spec, transforms=[])['image']
                else:
                    reconstructed = SpectrogramsDataModule.custom_augment_torchaudio(temp_spec, transforms=[transform])['image']
                
                reconstructed = reconstructed.float()
                reconstructed = reconstructed.numpy()[0]

                filename, ext = os.path.splitext(file_path)
                current_file_folder = os.path.basename(os.path.dirname(file_path))
                outfile = f"{os.path.basename(filename)}-{j}_{transform}{ratio:.2f}{ext}"
                outfile = os.path.join(out_folder, current_file_folder, outfile)
                os.makedirs(os.path.dirname(outfile), exist_ok=True)
                np.save(outfile, reconstructed)

    def noise(self, model, all_samples_paths, ratio=0.5, out_folder="", generation_count=10, device='cpu'):
        all_samples_paths = [x for x in all_samples_paths if 'noise' not in os.path.basename(x)]

        for i, sample_path in tqdm.tqdm(enumerate(all_samples_paths)):
            for j in range(generation_count):

                spectrogram = torch.tensor(self.load_sample(sample_path))
                q_t, q_b, i_t, i_b = self.encode(model, spectrogram, device=device)
                new_q_t = ratio*torch.randn_like(q_t) + q_t
                new_q_b = ratio*torch.randn_like(q_b) + q_b
                reconstructed = self.decode(model, new_q_t, new_q_b).cpu().numpy()[0][0]
                reconstructed = reconstructed[:,:-4]

                filename, ext = os.path.splitext(sample_path)
                outfile = f"{filename}-{j}_noise{ratio:.2f}{ext}"
                os.makedirs(out_folder, exist_ok=True)
                outfile = os.path.join(out_folder, os.path.basename(outfile))
                np.save(outfile, reconstructed)


    def interpolation(self, model, all_samples_paths, ratio=0.5, out_folder="", generation_count=10, device='cpu'):
        all_samples_paths = [x for x in all_samples_paths if 'interpolation' not in os.path.basename(x)]

        for i, sample_path in tqdm.tqdm(enumerate(all_samples_paths)):
            count = 0
            tmp_all_samples_paths = deepcopy(all_samples_paths)
            random.shuffle(tmp_all_samples_paths)

            for k, sample_path1 in enumerate(tmp_all_samples_paths):
                if count >= generation_count:
                    break
                if sample_path == sample_path1:
                        continue
                sample_class = sample_path.split(os.sep)[-2]
                sample_class1 = sample_path1.split(os.sep)[-2]
                if sample_class == sample_class1:
                    ratio = random.random() #Generate number in [0,1)
                    spectrogram = torch.tensor(self.load_sample(sample_path))
                    spectrogram1 = torch.tensor(self.load_sample(sample_path1))

                    q_t, q_b, i_t, i_b = self.encode(model.net, spectrogram, device=device)
                    q_t1, q_b1, i_t1, i_b1 = self.encode(model.net, spectrogram1, device=device)
                    new_q_t = (q_t1 - q_t)*ratio + q_t
                    new_q_b = (q_b1 - q_b)*ratio + q_b

                    reconstructed = self.decode(model.net, new_q_t, new_q_b).cpu().numpy()[0][0]
                    reconstructed = reconstructed[:,:-4]

                    filename, ext = os.path.splitext(sample_path)
                    outfile = f"{filename}-{k}_interpolation{ratio:.2f}-{os.path.basename(sample_path1)}{ext}"
                    np.save(outfile, reconstructed)
                    count += 1


    # @classmethod
    # def extrapolation(self, model, all_samples_paths, ratio=0.5, generation_count=10, device='cpu'):
    #     all_samples_paths = [x for x in all_samples_paths if 'extrapolation' not in os.path.basename(x)]

    #     count = 0
    #     for i, sample_path in tqdm.tqdm(enumerate(all_samples_paths)):
    #         for j in range(generation_count):
    #             for k, sample_path1 in enumerate(all_samples_paths):
    #                 if sample_path == sample_path1:
    #                     continue

    #                 ratio = random.rand(0,1)
    #                 spectrogram = torch.tensor(self.load_sample(sample_path))
    #                 spectrogram1 = torch.tensor(self.load_sample(sample_path1))

    #                 q_t, q_b, i_t, i_b = self.encode(model.net, spectrogram, device=device)
    #                 q_t1, q_b1, i_t1, i_b1 = self.encode(model.net, spectrogram1, device=device)
    #                 new_q_t = (q_t - q_t1)*ratio + q_t
    #                 new_q_b = (q_b - q_b1)*ratio + q_b

    #                 reconstructed = self.decode(model.net, new_q_t, new_q_b).cpu().numpy()[0][0]
    #                 reconstructed = reconstructed[:,:-4]

    #                 filename, ext = os.path.splitext(sample_path)
    #                 outfile = f"{filename}-{k}_extrapolation{ratio}-{os.path.basename(sample_path1)}{ext}"
    #                 np.save(outfile, reconstructed)

def update_model_keys(old_model:OrderedDict, key_to_replace:str='module.'):
    new_model = OrderedDict()
    for key,value in old_model.items():
        if key.startswith(key):
            new_model[key.replace(key_to_replace,'', 1)] = value
        else:
            new_model[key] = value
    return new_model

def load_model(model, model_path, device='cuda:0'):
    weights = torch.load(model_path, map_location='cpu')
    if 'model' in weights:
        weights = weights['model']
    if 'state_dict' in weights:
        weights = weights['state_dict']
    weights = update_model_keys(weights, key_to_replace='net.')
    model.load_state_dict(weights)
    model = model.eval()
    # model = model.to(device)
    return model

def main() -> None:
    args = parser.parse_args()
    # model = VQEngine.load_from_checkpoint(args.model_path).to(args.device)
    augmentations = Augmentations()
    model = VQVAE(in_channel=1)
    if args.model_path != "null":
        model_vqvae = load_model(model, args.model_path, device=device)

    all_samples_paths = glob.glob(args.data_paths)
    aug_methods_names = args.augmentations.split(',')
    for aug_method_name in aug_methods_names :
        if aug_method_name in Augmentations.specbase_augs:
            augmentations.specbase_augment(all_samples_paths=all_samples_paths, ratio=0.1, out_folder=args.out_folder, generation_count=int(args.num_samples), transform=aug_method_name, device=args.device, )

        elif aug_method_name in Augmentations.ecogen_augs:
            func = getattr(augmentations, aug_method_name)
            func(model=model, all_samples_paths=all_samples_paths, ratio=0.5, out_folder=args.out_folder, generation_count=int(args.num_samples), device=args.device, )
        else:
            raise NotImplementedError
    


if __name__ == "__main__":
    main()