from collections import OrderedDict
import time
import streamlit as st
import librosa
import librosa.display
import soundfile
import torch
from torch import nn
import numpy as np
import sys
import matplotlib.pyplot as plt
import cv2
# sys.path.append("../")
from networks.vqvae2 import VQVAE


st.title("VQ-VAE Visualization")

st.sidebar.title("Model Configuration")


models = {
    "null": "",
    # "taylor": "/home/future/Documents/runs_articfacts/birds-generation/outputs/2020-11-23/15-53-15/models-vqvae-v0.ckpt",
    # "british-birds": "/home/future/Documents/runs_articfacts/birds-generation/outputs/2021-02-13/10-04-49/models-vqvae.ckpt",
    "xeno-canto": "../paper-materials/epoch=5-step=30005.ckpt"
}
print(models)
model_path = st.sidebar.selectbox("Model File",
    list(models.keys()), index=1)


sr = st.sidebar.selectbox('Sample Rate', [16384,22500,44100], index=0)
device = st.sidebar.selectbox('device', ['cpu', 'cuda:0', 'cuda:1'], index=0)

n_fft = st.sidebar.selectbox('N_FFT', [512,1024,2048], index=1)
nb_seconds = st.sidebar.selectbox('Duration (s)', [1,2,4,8], index=2)
noise = st.sidebar.slider('Weight', min_value=0.0, max_value=10.0, value=0.5, step=0.01)

# model_path = st.text_input('Model Path', '/home/future/Documents/runs_articfacts/birds-generation/outputs/2020-11-23/15-53-15/models-vqvae-v0.ckpt')
# st.text(model_path)

infile_path = st.text_input('Input file', '')
all_input_files = [
    "",
    "/Users/test/Documents/Projects/Master/udem-birds/classes/AMGP_1(tu-tit)/108-14023579_14053660.wav",
    "/Users/test/Documents/Projects/Master/udem-birds/classes/WRSA_1(boingboingboing)/2-1380970_1481319.wav",
    "/Users/test/Documents/Projects/Master/taylor-dataset/media/Taylor/N_/14.wav",
    "/Users/test/Documents/Projects/Master/taylor-dataset/media/Taylor/NE_/1.-1.wav",
    "/Users/test/Documents/Projects/Master/taylor-dataset/media/Taylor/CaVi_/10.-10.wav",
    "/Users/test/Documents/Projects/Master/taylor-dataset/media/Taylor/BHGB_/10.-10.wav",
]
infile_path_select = st.selectbox('Input file', all_input_files, index=1)

if not infile_path:
    infile_path = infile_path_select
st.text(f"Selected file : {infile_path}")


def update_model_keys(old_model:OrderedDict, key_to_replace:str='module.'):
    new_model = OrderedDict()
    for key,value in old_model.items():
        if key.startswith(key):
            new_model[key.replace(key_to_replace,'', 1)] = value
        else:
            new_model[key] = value
    return new_model

@st.cache()
def load_model(model, model_path, device='cuda:0'):
    weights = torch.load(model_path, map_location='cpu')
    if 'model' in weights:
        weights = weights['model']
    if 'state_dict' in weights:
        weights = weights['state_dict']
    weights = update_model_keys(weights, key_to_replace='net.')
    model.load_state_dict(weights)
    model = model.eval()
    model = model.to(device)
    return model

# @st.cache()
def inference(model, img, noise=0.5, device='cuda:0'):
    img = img.to(device)
    quant_top, quant_bottom, diff, id_top, id_bottom = model.encode(img)
    
#     quant_top = nn.Dropout(0.2)(quant_top)
#     quant_bottom = nn.Dropout(0.2)(quant_bottom)    
# #     quant_top += noise*torch.randn(quant_top.shape).to(device)
# #     quant_bottom += noise*torch.randn(quant_bottom.shape).to(device)
    


    out = model.decode(quant_top, quant_bottom)
    return (quant_top.detach().cpu(),
           quant_bottom.detach().cpu(),
           id_top.detach().cpu().numpy(),
           id_bottom.detach().cpu().numpy(), 
           out.squeeze(0).detach().cpu())


def encode(model, img, device='cuda:0'):
    img = img.to(device)
    quant_top, quant_bottom, diff, id_top, id_bottom = model.encode(img)
    return (quant_top,
           quant_bottom,
           id_top,
           id_bottom)

def decode(model, quant_top, quant_bottom):
    return model.decode(quant_top, quant_bottom).detach()


@st.cache()
def load_audio(audio_path, sr=16384, seconds=4):
    audio, _sr = librosa.load(audio_path)
    audio = librosa.resample(audio, _sr, sr)
    audio = librosa.util.fix_length(audio, seconds)
    return audio

@st.cache()
def load_sample_spectrogram(audio_path, window_length=16384*4, sr=16384, n_fft=1024):
    audio = load_audio(audio_path, sr, window_length)
    
    
    features = librosa.feature.melspectrogram(y=audio, n_fft=n_fft)
    features = librosa.power_to_db(features)

    if features.shape[0] % 2 != 0: 
        features = features[1:, :]
    if features.shape[1] % 2 != 0:
        features = features[:, 1:]
        
    features_original = features

    features = np.expand_dims(features, 0)
    features = np.expand_dims(features, 0)

    features = torch.Tensor(features)
    return audio, features_original, features


@st.cache()
def spectrogram_to_audio(spec, sr=16384, n_fft = 1024):
    print(spec.shape)
#     spec = cv2.resize(spec, (129,128))
    spec = librosa.db_to_power(spec)
    audio = librosa.feature.inverse.mel_to_audio(spec, sr=sr, n_fft=n_fft)
    return audio


def play_audio(audio):
    st.text(f"Audio nb samples {len(audio)}")
    soundfile.write('./temp-audio.wav', audio, sr)
    st.text("Generate Input Audio (spec->audio)")
    with open('./temp-audio.wav', mode='rb') as reader:
        audio_bytes = reader.read()
        st.audio(audio_bytes, format="audio/wav")

        
if model_path and infile_path:
    st.write("Generation") 
    st.text("Loading models...")
    model_vqvae = VQVAE(in_channel=1)
    if model_path != "null":
        model_vqvae = load_model(model_vqvae, models[model_path], device=device)

    st.text("Loading inputs...")
    audio, img_orig, img = load_sample_spectrogram(infile_path, window_length=16384*nb_seconds, sr=sr, n_fft=n_fft)
    st.write(
        {
            "img.shape": img.shape,
            "audio length": len(audio)/sr*1.0,
        }
    )
    isPlaying = st.checkbox("Play Audio")
    # Play Audio
    if isPlaying:
        st.text("Input Audio (initial)")
        play_audio(audio)

#         with open(infile_path, mode='rb') as reader:
#             audio_bytes = reader.read()
#             st.audio(audio_bytes)
            
        st.text("Input Audio (spec->audio)")
        reconstructed_audio = spectrogram_to_audio(img[0][0].cpu().numpy())
        play_audio(reconstructed_audio)


    #Show input image
    fig, axs = plt.subplots(1, 2, figsize = (10,10))
    axs[0].imshow(img_orig)
    axs[0].set_xlabel('Input')

    st.text("Inference...")
    start_t = time.time()
    quant_top, quant_bottom, id_top, id_bottom, vq_vae_out = inference(model_vqvae, img, noise=noise, device=device)
    st.write(f"Execution time:{(time.time()-start_t )*1000} ms")
    
    out = vq_vae_out[0].numpy()
    #Show Reconstructed Image
    axs[1].imshow(out)
    axs[1].set_xlabel('Reconstructed')  
    st.pyplot(fig)
    
    #Show codebooks
    if st.checkbox("Show codebooks"):
        st.text({"id_top.shape": id_top.shape, "id_bottom.shape": id_bottom.shape})
        fig, axs = plt.subplots(1, 2, figsize = (15,20))
        axs[0].imshow(id_top[0], interpolation=None)
        axs[0].set_xlabel('Top Codebook')
        axs[1].imshow(id_bottom[0], interpolation=None)
        axs[1].set_xlabel('Bottom Codebook')  
        
        for i in range(len(id_top[0][0])):
            for j in range(len(id_top[0][1])):
                axs[0].text(i,j, id_top[0][j][i],ha='center',va='center')
        for i in range(len(id_bottom[0][0])):
            for j in range(len(id_bottom[0][1])):
                axs[1].text(i,j, id_bottom[0][j][i],ha='center',va='center', fontsize=6)
        st.pyplot(fig)

        
    #Play Reconstructed Audio
    if isPlaying:
        st.text("Reconstructing audio...")
        st.text(out.shape)
        reconstructed_audio = spectrogram_to_audio(out)
        play_audio(reconstructed_audio)

            
        if st.checkbox("Show waveforms"):
            fig, axes = plt.subplots(2,1, figsize=(5,5))
            librosa.display.waveplot(audio, sr=sr, ax=axes[0])
            axes[0].set(title='Input')
            axes[0].label_outer()
            librosa.display.waveplot(reconstructed_audio, sr=sr, ax=axes[1])
            axes[1].set(title='Reconstructed')
            axes[1].label_outer()
            st.pyplot(fig)
        
    ## Interpolation
    st.title("Interpolation")
    col1, col2 = st.beta_columns(2)
    infile_path = col1.selectbox('A', all_input_files, index=1)
    infile_path1 = col2.selectbox('B', all_input_files, index=1)
    
    
    audio, img_orig, img = load_sample_spectrogram(infile_path, window_length=16384*nb_seconds, sr=sr, n_fft=n_fft)
    audio1, img_orig1, img1 = load_sample_spectrogram(infile_path1, window_length=16384*nb_seconds, sr=sr, n_fft=n_fft)


    q_t, q_b, i_t, i_b = encode(model_vqvae, img, device=device)
    q_t1, q_b1, i_t1, i_b1 = encode(model_vqvae, img1, device=device)
    
    new_q_t, new_q_b = (q_t1 - q_t)*noise + q_t, (q_b1 - q_b)*noise + q_b
    out = decode(model_vqvae, new_q_t, new_q_b).cpu().numpy()
    
    new_q_t = new_q_t.detach().cpu()
    new_q_b = new_q_b.detach().cpu()

    fig, axs = plt.subplots(1, 3, figsize = (10,10))
    axs[0].imshow(img_orig)
    axs[0].set_xlabel('Input A')
    
    axs[1].imshow(img_orig1)
    axs[1].set_xlabel('Input B')
    
    axs[2].imshow(out[0][0])
    axs[2].set_xlabel('Interpolated')
    st.pyplot(fig)
    reconstructed_audio = spectrogram_to_audio(out[0][0])
    play_audio(reconstructed_audio)
    
    
    #TODO: Add noise augmentation.