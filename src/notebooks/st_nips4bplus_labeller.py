from email.mime import audio
import os
import tempfile
import time
import glob
import librosa
from tqdm import tqdm
import streamlit as st
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)


def load_annotation(annotation_file):
    outs = []
    if os.path.isfile(annotation_file) is False:
        return []
    with open(annotation_file, 'r') as fs:
        try:
            data = fs.read().splitlines()
            for row in data:
                if not row:
                    continue
                start_t, duration, classname = row.split(',')
                start_t, duration = float(start_t), float(duration)
                if duration < 0.25:
                    continue
                outs.append([start_t, duration, classname])
        except Exception as e:
            print(e, "Error while processing", annotation_file, print(data))
            return []
    return outs

def play_audio_notebook(y, sr):
    return IPython.display.display(IPython.display.Audio(y, rate=sr))

def play_audio(y, sr, st_temp):
    file_object, file_path = tempfile.mkstemp(suffix=".wav")
    librosa.output.write_wav(file_path, y, sr)
    with open(file_path, "rb") as fs:
        audio_bytes = fs.read()
        st_temp.audio(audio_bytes, format="audio/wav")
    os.remove(file_path)

def f_high(y,sr):
    b,a = signal.butter(10, 2000/(sr/2), btype='highpass')
    yf = signal.lfilter(b,a,y)
    return yf

def compute_spectrogram(y, sr):
    D = librosa.stft(y)
    D = librosa.amplitude_to_db(D, ref=np.max)
    return D

def compute_mel_spectrogram(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    D = librosa.power_to_db(S, ref=np.max) 
    return D


sidebar = st.sidebar


sidebar.title("Parameters")
root_folder ="/Users/test/Documents/Projects/Master/nips4bplus"
audio_paths = sidebar.text_input("Audio Path", f"{root_folder}/raw_audio/train")
outfolder = sidebar.text_input("Output Folder", f"{root_folder}/raw_audio/new_labels_temp/")
SR = sidebar.select_slider("Sample Rate", options=[16386, 22050, 44100], value=22050)

all_audios_paths = glob.glob(f"{audio_paths}{os.sep}*.wav")
# all_audios_paths = [x for x in all_audios_paths if "321" in x]
cpt = sidebar.number_input("Sample Index", value=0, min_value=0, max_value=len(all_audios_paths))
sidebar.write(f"Step{cpt}/{len(all_audios_paths)}")

st.title("Dataset size: {}".format(len(all_audios_paths)))
st.text(os.path.basename(all_audios_paths[cpt]))
audio_path = all_audios_paths[cpt]
y, sr = librosa.load(audio_path, sr=SR)


fig_, ax_ = plt.subplots(nrows=2, figsize=(20,10))
spectrogram = compute_spectrogram(y, SR)
librosa.display.specshow(spectrogram, x_axis='time', y_axis='log', sr=SR, ax=ax_[0])
mel_spectrogram = compute_mel_spectrogram(y, SR)
librosa.display.specshow(mel_spectrogram, x_axis='time', y_axis='mel', ax=ax_[1])
st.pyplot()
play_audio(y, SR, st)


#Load all annotations. Pick a color per class, Show the annotations timestamples on the waveform. Extract the crop
name_token = audio_path.split('_')[-1].replace("trainfile", "train")
annotation_file = f"annotation_{name_token}".replace(".wav", ".csv") 
annotation_file = os.path.join(root_folder, "raw_audio", "new_annotations", annotation_file)
annotations = load_annotation(annotation_file)
st.write(f"Annotation File: {os.path.basename(annotation_file)} -- {len(annotations)}")

start_times = []
end_times = []
crops_specgrams = []
y_samples = []

os.makedirs(outfolder, exist_ok=True)
if annotations:
    fig, ax = plt.subplots(nrows=len(annotations), figsize=(1,1))
    if isinstance(ax, np.ndarray) is False:
        ax = [ax]
    
    for j in range(len(annotations)):
        start_times.append(annotations[j][0])
        end_times.append(annotations[j][0]+ annotations[j][1])

        start_, end_ = librosa.time_to_samples([max(0, start_times[-1]-0.01), min(end_times[-1]+0.01, len(y))], sr=SR)
        y_sample = y[start_:end_]
        mel_sample = compute_mel_spectrogram(y_sample, SR)
        crops_specgrams.append(dict(data=mel_sample, x_axis='time', y_axis='mel', ax=ax[j]))
        y_samples.append(y_sample)


    # ax_[-1].vlines(start_times, 0, 1, color='r', alpha=0.9, linestyle='--', label='start')
    # ax_[-1].vlines(end_times, 0, 1, color='b', alpha=0.9, linestyle='--', label='end')
    # ax_[-1].axis('tight')
    # ax_[-1].legend(frameon=True, framealpha=0.75)

    # form = st.form("my_form")
    form = st
    columns = form.columns(len(annotations))
    choices = [-1]*len(annotations)
    outfile = os.path.join(outfolder, "revised-"+os.path.basename(annotation_file))
    if os.path.isfile(outfile):
        form.write(f"{outfile} already exists.")

    #reate bookmark for image of interest. Spectrograms that don't have a ny annotations.
    for j, col in enumerate(columns):
        #show crop spectrogram
        ax[j].axis('off')
        img = librosa.display.specshow(**crops_specgrams[j])
        #show choice selection
        choices[j] = col.radio(f"crop:{j}", options=["Yes", "No"], index=0)
        # col.pyplot()
        play_audio(y_samples[j], SR, col)
        with open(outfile, 'w') as fs:
            for k, choice in enumerate(choices):
                row = annotations[k]
                if choice=="Yes":
                    fs.write(f"{row[0]}, {row[1]}, {row[2]}\n")
    
    st.pyplot()
    # form.form_submit_button('Submit')

else:
    st.write("No annotations found.")
    logfile = os.path.join(outfolder, "bad_files.log")
    with open(logfile, 'a') as fs:
        fs.write(audio_path+"\n")


# if st.button("Next"):
#     cpt = (cpt+1)%len(all_audios_paths)
# if st.button("Prev"):
#     cpt -= 1
#     if cpt < 0:
#         cpt = len(all_audios_paths) -1
    
    # break