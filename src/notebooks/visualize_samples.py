import os
import time
import sys
sys.path.append("../")
from dataloader import AudioDataset
import torch
from natsort import natsorted
import random
import argparse
from glob import glob
import streamlit as st
import pandas as pd
import numpy as np
import librosa
import librosa.display
import librosa.feature
import matplotlib.pyplot as plt
from PIL import Image
import sklearn

random.seed(7)

parser = argparse.ArgumentParser(description="Visualizer for audio samples")
parser.add_argument("--root_path", type=str, default="/home/future/Documents/Datasets/BirdsSong/Priority")
parser.add_argument("--samples", type=int, default=5)


@st.cache()
def load_metadata(path):
    metadata_file = pd.read_csv(path)
    return metadata_file

def show_spectrogram2(*args, **kwargs):
    plt.figure(figsize=(15, 5))
    librosa.display.specshow(*args, **kwargs)


def show_spectogram(y, sr):
    try:
        plt.figure(figsize=(30,20))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        plt.subplot(2,1,1)
        librosa.display.specshow(D, y_axis='linear')
        # plt.colorbar(format='%+2.0f dB')
        plt.subplot(2,1,2)
        librosa.display.waveplot(y, sr=sr)
        plt.title('Linear-frequency power spectrogram')
        st.pyplot()
    except Exception as e:
        pass


def tryint(x):
    try:
        int(x)
        return True
    except ValueError:
        return False

if __name__ == "__main__":
    args = parser.parse_args()

    """
    Side bar definition
    """
    # Adds a checkbox to the sidebar
    audio_root = st.sidebar.text_input("Audio Folder")

    st.title("UdeM Birds Generated Audio- VQ-VAE2")

    image = Image.open("./vq-vae2.jpeg")
    st.image(image, caption='Conditional Gan')
    st.write("https://arxiv.org/pdf/1411.1784.pdf")
    # generated_samples_path = "/home/future/Documents/Projects/AudioGeneration/src/WaveGAN-pytorch/artifacts/20191125083141"
    # generated_samples_path = "/media/future/Rapido/udem-birds/classes"
    try:
        audio_samples_path = audio_root
        if audio_samples_path == "":
            raise FileNotFoundError
        st.write(audio_samples_path)
        subfolders = [os.path.basename(x) for x in glob(f"{audio_samples_path}/*") if os.path.isdir(x)]
        subfolders = natsorted(subfolders)[::-1]
        subfolder = st.sidebar.radio('Experiments', subfolders)

        # steps  = [int(os.path.basename(x)) for x in glob(f"{audio_samples_path}/{subfolder}/*") if tryint(os.path.basename(x))]
        # steps.sort()
        # interval = steps[1] - steps[0] if len(steps) > 2 else steps[0]
        # print(interval)

        # step = st.sidebar.slider('Epochs', min_value=min(steps), max_value=max(steps), step=interval)

        classe = st.sidebar.radio('Classes', natsorted([os.path.basename(x) for x in glob(f"{audio_samples_path}/{subfolder}/out/[0-9]*")]))
        audios = glob(f"{audio_samples_path}/{subfolder}/out/*.out")
        audios =  natsorted(audios)
        print("Total samples", len(audios))
        if len(audios) > 0:
            st.header("Generated Audio: {}  -- epoch:{}".format(subfolder,0))
        if os.path.isfile(f"{audio_samples_path}/{subfolder}/out/sample{classe}"):
            samples = torch.load(f"{audio_samples_path}/{subfolder}/out/sample{classe}")
        outs = torch.load(f"{audio_samples_path}/{subfolder}/out/{classe}")
        print(len(outs))
        # st.write(samples.shape)
        for j in range(len(outs)):
            if j < 3 :
                pass
            else:
                # break
                pass
            sample = samples[j].cpu().numpy()
            out = outs[j].cpu().numpy()

            show_spectrogram2(sample[0], x_axis='time', y_axis='mel')
            st.pyplot()
            show_spectrogram2(out[0], x_axis='time', y_axis='mel')
            st.pyplot()
            # st.text(real.shape)
            # break
            start_time = time.time()
            if sample.shape[0] == 2:
                sample[0] = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 250.0)).fit_transform(sample[0])
                audio = AudioDataset.specgram_to_audio(np.transpose(sample, (1,2,0)), resize=(65,513))
                st.write(" Specgram Execution time", time.time() - start_time)
            elif sample.shape[0] == 1:
                sample[0] = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 250.0)).fit_transform(sample[0])
                audio = AudioDataset.mel_spectrogram_to_audio(sample[0], resize=False)
                st.write(" MelSpectrogram Execution time", time.time() - start_time)

            st.text(f"{audio_samples_path}/{subfolder}/out/{classe}")
            start_time = time.time()
            if out.shape[0] == 2:
                out[0] = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 250.0)).fit_transform(out[0])
                audio_out = AudioDataset.specgram_to_audio(np.transpose(out, (1,2,0)), resize=(65,513))
                st.write(" Specgram Execution time", time.time() - start_time)
            elif out.shape[0] == 1:
                out[0] = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 250.0)).fit_transform(out[0])
                audio_out = AudioDataset.mel_spectrogram_to_audio(out[0], resize=False)
                st.write(" MelSpectrogram Execution time", time.time() - start_time)

            st.write(audio.shape, audio_out.shape)
            librosa.output.write_wav("./temp.wav", audio, 16384)
            librosa.output.write_wav("./temp_out.wav", audio_out, 16384)

            # y, sr = librosa.load(path, sr=16000)
            # show_spectogram(y, sr)
            with open("./temp.wav", "rb") as fs:
                audio_bytes = fs.read()
                st.audio(audio_bytes, format="audio/wav")
            with open("./temp_out.wav", "rb") as fs:
                audio_bytes = fs.read()
                st.audio(audio_bytes, format="audio/wav")

    except FileNotFoundError as e:
        pass
