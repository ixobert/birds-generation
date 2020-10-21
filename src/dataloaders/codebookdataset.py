import lmdb
import pickle
import torch

class LMDBDataset():
    def __init__(self, data_path):
        self.data_path = data_path
        self.env = lmdb.open(self.data_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            raise IOError('Cannot open lmdb dataset', self.data_path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = str(index).encode('utf-8')

            row = pickle.loads(txn.get(key))

        #TODO: Return nd.array instead torch.Tensor
        return row.top, row.bottom, row.filename



#TODO: Add scheduler to the trainer Engine.