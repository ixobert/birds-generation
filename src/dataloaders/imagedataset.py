import os
import natsort
import uuid
import random
import pickle
import cv2
import numpy as np
import torch


class ImageDataset():
    def __init__(self, root_dir, data_path, classes_name, return_tuple=True):
        self.data_path = data_path
        self.root_dir = root_dir
        self.classes_name = classes_name
        self.return_tuple = return_tuple
        print("Data initialization")

        self.data_paths = []
        with open(self.data_path, 'r') as reader:
            data = reader.read().splitlines()
            print("All paths", len(data))
            for d in data:
                for cls in self.classes_name:
                    label = self.extract_label(d)
                    if cls == label:
                        self.data_paths.append(d)
            self.data_paths = [os.path.join(self.root_dir, x) for x in self.data_paths]
        
        self.data = self.get_cached_dataset(files_path=self.data_paths, load_function=cv2.imread)
        print("Loaded data from cache", len(self.data), len(self.data_paths))

        #Only keep samples that have the class considered for the experiment.
        self.data = [ x for x in self.data if True in [class_ in x[0] for class_ in self.classes_name] ]
        random.shuffle(self.data)
            # self.data = random.sample(self.data, k=min(len(self.data),10))  # For debugging
        print("Data initialization done", len(self.data))
        if len(self.data) < 1:
            print("Empty dataset")
            raise ValueError

    def __len__(self):
        return len(self.data)

    def extract_label(self, filepath):
        return filepath.split('/')[-2]

    @classmethod
    def get_cached_dataset(cls, files_path:str, cache_folder:str ='/tmp/cached_dataset', load_function=None) -> list:
        """
            This Function build (or load if it already exists) and return it as an output
        """
        sorted_paths = natsort.natsorted(files_path)
        files_concat = "-".join(sorted_paths)
        files_hash = str(uuid.uuid3(uuid.NAMESPACE_OID, files_concat))
        cache_file_path = os.path.join(cache_folder, files_hash)
        os.makedirs(cache_folder, exist_ok=True)
        if os.path.isfile(cache_file_path):
            print("Load cached Data", files_hash)
            with open(cache_file_path, 'rb') as reader:
                data = pickle.load(reader)
        else:
            print("Create cached Data", files_hash)
            data = [(file_path, load_function(file_path)) for file_path in files_path]
            with open(cache_file_path, 'wb') as writer:
                pickle.dump(data, writer)
        return data


    def __getitem__(self, idx):
        file_path, img = self.data[idx]
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, 0)
        label_name = self.extract_label(file_path)
        label = self.classes_name.index(label_name)
        label = torch.nn.functional.one_hot(torch.tensor(label), len(self.classes_name))

        if self.return_tuple:
            return torch.Tensor(img), label, file_path        

        return {
            "x": torch.Tensor(img),
            "y": label,
            # "audio": audio,
            # "label": label,
        }
