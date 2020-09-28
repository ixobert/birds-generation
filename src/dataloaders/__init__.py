import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
try:
    from audiodataset import AudioDataset
except ModuleNotFoundError:
    from .audiodataset import AudioDataset



class SpectrogramsDataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.config = kwargs.get('config')
        self.batch_size = self.config.get('batch_size', 2)

    def setup(self, stage=None):
        #################
        #TODO: To delete: this code makes my life easier during development
        import sys
        platform = sys.platform.lower()
        print(f"Running on : {platform}")
        if platform == 'darwin':
            root_ = "/Users/test/Documents/Projects/Master/"
            self.config['root_dir'] = os.path.join(root_, "udem-birds/classes")
            self.config['train_path'] = os.path.join(root_, "udem-birds/samples/train_list.txt")
        #################
        self.dataset = AudioDataset(data_path=self.config['train_path'], root_dir=self.config['root_dir'], classes_name=self.config['classes_name'], sr=self.config['sr'], window_length=self.config['sr']*4, spec=True, resize=False, return_tuple=True, use_spectrogram=self.config.get('use_mel', False))

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.config['num_workers'])

if __name__ == "__main__":
    root_ = "/Users/test/Documents/Projects/Master/"
    # root_ = "/media/future/Rapido/"
    root_dir = os.path.join(root_, "udem-birds/classes")
    config = dict(
        root_dir = root_dir,
        train_path = os.path.join(root_, "udem-birds/samples/train_list.txt"), 
        classes_name=['AMGP_1(tu-tit)', 'AMGP_2(tuuut)', 'WRSA_1(boingboingboing)'],
        sr=16384,
        window_length=16384*4,
        spec=True,
        use_mel=True,
        num_workers=4,
    )
    module = SpectrogramsDataModule(config=config)
    module.setup()
    for data in module.train_dataloader():
        print(type(data))