import logging
import os
import pickle
import natsort
import uuid
# import wavefile
import librosa
import librosa.feature
import torch
from glob import glob
import numpy as np
import random
import cv2
import sklearn.preprocessing
import warnings
warnings.filterwarnings('ignore', 'PySoundFile failed. Trying audioread instead.')
try:
    from helpers import specgram, ispecgram
except ModuleNotFoundError:
    from .helpers import specgram, ispecgram


class AudioDataset():
    def __init__(self, root_dir, data_path, classes_name, sr=16000, window_length=16384, spec=False, resize=True, return_tuple=False, return_tuple_of3=True, use_spectrogram=False, use_cache=True, use_rgb=False):
        self.data_path = data_path
        self.root_dir = root_dir
        self.classes_name = classes_name
        self.sr = sr
        self.window_length = window_length
        self.spec = spec
        self.precomputed_available = False
        self.resize = resize
        self.return_tuple = return_tuple
        self.return_tuple_of3 = return_tuple_of3
        self.use_spectrogram = use_spectrogram
        self.use_cache = use_cache
        self.use_rgb = use_rgb
        if self.use_spectrogram and not self.spec:
            print("Overriding spec variables because use_spectrogram is true")
            self.spec = True
        print("Data initialization")

        self.data_paths = []
        with open(self.data_path, 'r') as reader:
            data = reader.read().splitlines()
            print("All paths", len(data))
            for d in data:
                for cls in self.classes_name:
                    if cls in d:
                        self.data_paths.append(d)
            self.data_paths = [os.path.join(self.root_dir, x).strip() for x in self.data_paths]
        
        if self.use_cache:
            self.data = self.get_cached_dataset(files_path=self.data_paths)
            print("Loaded data from cache", len(self.data), len(self.data_paths))
        else:
            self.data = [(file_path, None) for file_path in self.data_paths]
            print("Not using cache", len(self.data), len(self.data_paths))

        #Only keep samples that have the class considered for the experiment.
        self.data = [ x for x in self.data if True in [class_ in x[0] for class_ in classes_name] ]
        random.shuffle(self.data)
            # self.data = random.sample(self.data, k=min(len(self.data),10))  # For debugging
        logging.info(f"Data initialization done {len(self.data)}")
        if len(self.data) < 1:
            logging.info("Empty dataset")
            raise ValueError

    def __len__(self):
        return len(self.data)

    @classmethod
    def get_cached_dataset(cls, files_path:str, cache_folder:str ='/tmp/cached_dataset') -> list:
        """
            This Function build (or load if it already exists) and return it as an output
        """
        sorted_paths = natsort.natsorted(files_path)
        files_concat = "-".join(sorted_paths)
        files_hash = str(uuid.uuid3(uuid.NAMESPACE_OID, files_concat))
        cache_file_path = os.path.join(cache_folder, files_hash)
        os.makedirs(cache_folder, exist_ok=True)
        if os.path.isfile(cache_file_path):
            logging.info(f"Load cached Data {cache_file_path}")
            with open(cache_file_path, 'rb') as reader:
                data = pickle.load(reader)
        else:
            logging.info(f"Create cached Data {files_hash}")
            data = []
            for file_path in files_path:
                audio, _sr = librosa.load(file_path)
                data.append((file_path, audio))
            with open(cache_file_path, 'wb') as writer:
                pickle.dump(data, writer)
        return data


    def add_pad(self, sample, window_length):
        pad_length = window_length - len(sample)
        left_pad = pad_length // 2
        right_pad = pad_length - left_pad
        return np.pad(sample, (left_pad, right_pad), mode='constant')

    @classmethod
    def preprocess(self, audio):
        max_magnitude = np.max(np.abs(audio))
        if max_magnitude > 1:
            return audio / max_magnitude
        return audio

    @classmethod
    def audio_to_specgram(self, audio, resize=True):
        features = specgram(audio, n_fft=1024)
        if resize:
            features = cv2.resize(features,(32,256))
        if features.shape[0] % 2 != 0:
            features = features[1:, :, :]
        if features.shape[1] % 2 != 0:
            features = features[:, 1:, :]
        
        features = np.transpose(features, (2, 0, 1))
        return features

    @classmethod
    def audio_to_melspectrogram(self, audio, resize=True):
        "This function only works for image with one channel"
        features = librosa.feature.melspectrogram(y=audio, n_fft=1024)
        features = librosa.power_to_db(features)
        if resize:
            features = cv2.resize(features,(32,256))
        #If height is odd, skip the first column
        if features.shape[0] % 2 != 0: 
            features = features[1:, :]
        #If row is odd, skip the first row
        if features.shape[1] % 2 != 0:
            features = features[:, 1:]
        features = np.expand_dims(features, 0)
        return features


    @classmethod
    def mel_spectrogram_to_audio(self, spectrogram, resize=True):
        if resize:
            if isinstance(resize, set):
                spectrogram = cv2.resize(spectrogram, resize)
            else:
                spectrogram = cv2.resize(spectrogram,(64,128))
        print("Mel shape", spectrogram.shape)
        spectrogram = librosa.db_to_power(spectrogram)
        s = librosa.feature.inverse.mel_to_stft(spectrogram)
        y = librosa.griffinlim(s)
        return y


    @classmethod
    def specgram_to_audio(self, specgram, resize=True):
        if resize:
            if isinstance(resize, set):
                specgram = cv2.resize(specgram, resize)
            else:
                specgram = cv2.resize(specgram,(129,513))

        # if specgram.shape[0] % 2 == 0:
        #     specgram= np.pad(specgram, ((0,1),(0,0),(0,0)))
        # if specgram.shape[1] % 2 == 0:
        #     specgram= np.pad(specgram, ((0,0),(0,1),(0,0)))
        return ispecgram(specgram, n_fft=1024)


    def load_audio(self,file_path, sr, window_length=0):
        audio, _sr = librosa.load(file_path, sr=sr)
        if _sr != self.sr:
            audio = librosa.resample(audio, _sr, sr)

        if self.window_length and len(audio) >= self.window_length:
            audio = audio[0:self.window_length]
        else:
            audio = librosa.util.fix_length(audio, self.window_length)
        return audio

    def __getitem__(self, idx):
        file_path, audio = self.data[idx]
        try:
            if audio is None:
                if file_path.endswith('.npy'):
                    features = np.load(file_path) 
                    features= np.expand_dims(features,0)
                else:
                    audio = self.load_audio(file_path, self.sr, self.sr*4)
                
            if self.spec:
                if self.use_spectrogram:
                    features = self.audio_to_melspectrogram(audio, resize=self.resize)
                else:
                    features = self.audio_to_specgram(audio, resize=self.resize)
                # features = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(features)
            else:
                features = audio
            if 'udem' in file_path:
                label_name = file_path.split('/')[-2]
            # elif 'nsynth' in file_path:
                # label_name = os.path.basename(file_path).split('_')[0]
            elif 'flute' in file_path:
                label_name = os.path.basename(file_path).split('-')[1]
            else:
                label_name = file_path.split('/')[-2]
            label = self.classes_name.index(label_name)
            one_hot_label = np.zeros(len(self.classes_name))
            one_hot_label[label] = 1
            # print(one_hot_label, label, label_name)


            features = torch.tensor(features)
            label = torch.tensor(label)
            if self.return_tuple:
                if self.return_tuple_of3:
                    return features, label, file_path
                else:
                    ##TODO: fix make special case for image classifier.
                    if self.use_rgb:
                        features = np.concatenate(3*[features]) #Single channel to 3 channel
                    return features, label
            return {
                "x": features,
                "y": torch.Tensor(np.expand_dims(one_hot_label, 0)).long(),
                # "audio": audio,
                # "label": label,
            }
        except Exception as e:
            logging.info(f"Error {e} on file: {file_path}")
            if self.return_tuple_of3:
                return None, None, None
            else:
                return None, None


if __name__ == "__main__":
    # root_dir = "/media/future/Rapido/nsynth_dataset/nsynth-train/audio"
    # dataset_path = "/media/future/Rapido/nsynth_dataset/flute-acoustic-56-127-train_val.txt"
    # classes = [x for x in os.listdir(f"{root_dir}/") if 'background' not in x]
    # classes = ['flute', 'bass', 'brass', 'organ', 'mallet', 'guitar']
    # classes = [f"0{x}" for x in range(55, 69)]
    root_ = "/Users/test/Documents/Projects/Master/"
    # root_ = "/media/future/Rapido/"
    root_dir = os.path.join(root_, "udem-birds/classes")
    dataset_path = os.path.join(root_, "udem-birds/samples/train_list.txt")
    classes = ['AMGP_1(tu-tit)', 'AMGP_2(tuuut)', 'WRSA_1(boingboingboing)']
    print(classes)
    dataset = AudioDataset(root_dir, dataset_path, classes,
                           sr=16384, window_length=16384*4, spec=True, use_spectrogram=True)
    print("Dataset size: {}".format(len(dataset)))
    for data in dataset:
        print(data['x'].shape, data['y'].shape)
        pass
