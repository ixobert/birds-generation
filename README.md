# ECOGEN: Bird Sounds Generation using Deep Learning (Work in Progress)

This repository contains the code for the paper [ECOGEN: Bird Sounds Generation using Deep Learning]($PAPER_LINK). 
The paper proposes a novel method for generating bird sounds using deep learning by leveraging VQ-VAE2 network architecture.
The proposed method is able to generate bird sounds that aims to increase the dataset size for bird sound classification tasks.



## Dataset
The dataset used in this paper is the Xeno-Canto dataset from Kaggle. The dataset can be downloaded from [Part 1](https://www.kaggle.com/rohanrao/xeno-canto-bird-recordings-extended-a-m) and [Part 2](https://www.kaggle.com/rohanrao/xeno-canto-bird-recordings-extended-n-z).

## Model Checkpoint
MOdel checkpoints can be found in the [OSF Link](https://doi.org/10.17605/OSF.IO/YQDJ9) folder.
## Requirements
The code is tested on Python 3.7.7 and PyTorch 1.13.1. The required packages can be installed using the following command:
```
git clone https://github.com/ixobert/birds-generation
cd ./birds-generation/
pip install -r requirements.txt
```

## Preprocessing
The preprocessing steps are as follows:
1. Convert the audio files to mono channel
2. Resample the audio files to 22050 Hz
3. Trim the audio files to 5 seconds


## Usage
We heavily used Hydra to manage the configuration files. The configuration files can be found in the `src/configs` folder. See the [Hydra documentation](https://hydra.cc/docs/intro) for more details.

### ECOGEN Training
The ECOGEN training code is inspired from the [VQ-VAE2 implementation]($PAPER_LINK).
The training code can be found in the `src` folder.
The code expects the dataset to be in the following format:
```
./birds-songs/dataset/train.txt|test.txt
```

The train,test and validation text files contains the path to the audio files. See below an example of a train.txt file:

```
birds-song/1.wav
birds-song/2.wav
birds-song/3.wav
```


To train the ECOGEN model, run the following command:
```
python ./src/train_vqvae.py  dataset="xeno-canto" mode="train" lr=0.00002 nb_epochs=25000 log_frequency=1 dataset.batch_size=420 dataset.num_workers=8 run_name="ECOGEN Training on Xeno Canto"  tags=[vq-vae2,xeno-canto] +gpus=[1] debug=false
```

#### Sample Generation
The current version of ECOGEN supports 2 types of augmentation, interpolation and noise.
To generate th  e bird sounds, run the following command:
#### Noise augmentation
```
python ./src/generate.py  --ckpt <path_to_ckpt> --num_samples 100 --noise 0.1 --gpus 0
```
### Interpolation augmentation
```
python ./src/generate.py  --ckpt <path_to_ckpt> --num_samples 100 --noise 0.1 --gpus 0
```