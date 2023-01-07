import os
import random
random.seed(42)
import cv2
import torch
import torchvision
import torchaudio

class RawAudioDataset:
    def __init__(self, root_dir, data_path, classes_name, sr=16000, window_length=16384, use_spectrogram=False, transform=None, *args, **kwargs) -> None:
        self.data_path = data_path
        self.root_dir = root_dir
        self.classes_name = classes_name
        self.sr = sr
        self.window_length = window_length
        self.use_spectrogram = use_spectrogram
        self.transform = transform
        self.data_paths = []
        self.data = []

        with open(self.data_path) as reader:
            file_paths= reader.read().splitlines()
            print("All file paths: ", len(file_paths))
            for file_path in file_paths:
                file_path = os.path.join(self.root_dir, file_path)
                file_path = "".join([os.path.splitext(file_path)[0], ".wav"])
                file_path = os.path.abspath(file_path)
                if not os.path.isfile(file_path):
                    continue
                for cls in self.classes_name:
                    if cls in file_path:
                        self.data_paths.append([file_path, cls])
                        break
    
        random.shuffle(self.data_paths)
        print("Data paths: ", len(self.data_paths))
        for audio_path, cls  in self.data_paths:
            audio_sample, orig_sr = torchaudio.load(audio_path)
            if self.use_spectrogram:
                audio_sample = torchaudio.transforms.Spectrogram(n_fft=1024)(audio_sample)
            self.data.append([audio_sample, self.classes_name.index(cls)])

    def __len__(self,):
        """Return the number of data."""
        return len(self.data)
    

    def __getitem__(self, idx):
        """Return one data pair (audio and label)."""
        audio, label = self.data[idx]
        label = torch.tensor(label, dtype=torch.long)
        if self.transform:
            audio = self.transform(audio)
        audio = audio.expand(3, audio.shape[1], audio.shape[2])
        return audio, label

