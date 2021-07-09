import sys
import os
import wandb
from typing import List
import logging
import librosa
os.environ['HYDRA_FULL_ERROR'] = '1'
import hydra
from omegaconf import DictConfig
from sklearn.metrics import classification_report
import torch
import flash
from flash.vision import ImageClassificationData, ImageClassifier
from pytorch_lightning.loggers import WandbLogger as Logger
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image
try:
    import networks
    from dataloaders import SpectrogramsDataModule
    from dataloaders.audiodataset import AudioDataset
except ImportError:
    from src import networks
    from src.dataloaders.audiodataset import AudioDataset
    from src.dataloaders import SpectrogramsDataModule


def path_to_label(path:str, format:str = 'index', classes:List=None) -> int:
    """Extract the class name from file path and return the corresponding class_idx in classes.

    Args:
        path (str): [description]
        format(str): 'index' or 'name'
        classes (List): [description]

    Returns:
        [int]: Class index
    """
    class_name = path.split(os.sep)[-2]
    try:
        if format == 'index':
            return classes.index(class_name)
        else:
            return class_name
    except ValueError:
        return -1


def load_filepaths(infile:str) -> List[str]:
    with open(infile, mode='r') as reader:
        return reader.read().splitlines()

def load_sample(file_path, nb_second=4, sr=16384, resize=False):
    window_length = sr*nb_second
    audio, _sr = librosa.load(file_path, sr=sr)
    if _sr != sr:
        audio = librosa.resample(audio, _sr, sr)
    if len(audio) >= window_length:
        audio = audio[0:window_length]
    else:
        audio = librosa.util.fix_length(audio, window_length)
    features = AudioDataset.audio_to_melspectrogram(audio, resize=resize)
    features = torch.from_numpy(features)
    return features



def get_data(cfg):
    train_filepaths = load_filepaths(cfg['dataset']['train_path'])
    val_filepaths   = load_filepaths(cfg['dataset']['val_path'])
    test_filepaths  = load_filepaths(cfg['dataset']['test_path'])
    train_filepaths = [os.path.join(cfg['dataset']['root_dir'], x) for x in train_filepaths]
    val_filepaths   = [os.path.join(cfg['dataset']['root_dir'], x) for x in val_filepaths]
    test_filepaths  = [os.path.join(cfg['dataset']['root_dir'], x) for x in test_filepaths]
    CLASSES_NAMES = cfg['dataset']['classes_name']
    datamodule = ImageClassificationData.from_filepaths(
        train_filepaths= train_filepaths,
        train_labels   = [path_to_label(x, format="index", classes=CLASSES_NAMES) for x in train_filepaths],
        val_filepaths  = val_filepaths,
        val_labels     = [path_to_label(x, format="index", classes=CLASSES_NAMES) for x in val_filepaths],
        test_filepaths = test_filepaths,
        test_labels    = [path_to_label(x, format="index", classes=CLASSES_NAMES) for x in test_filepaths],
        batch_size     = cfg['dataset']['batch_size'],
        num_workers    = cfg['dataset']['num_workers'],
        loader = load_sample,
    )
    return datamodule


def evaluate_model(model, dataloader):
    targets = []
    predictions = []
    for features, labels in dataloader():
        out = model.predict(features)
        predictions.extend(out)
        targets.extend(labels.numpy())
    return predictions, targets


@hydra.main(config_path="configs", config_name="train_classifier")
def main(cfg: DictConfig) -> None:
    current_folder = os.getcwd()
    logging.info(f"Current Folder:{current_folder}")

    if cfg.get('debug', False):
        logger = Logger(project=cfg['project_name'], name=cfg['run_name'], tags=cfg['tags']+["debug"])
    else:
        logger = Logger(project=cfg['project_name'], name=cfg['run_name'], tags=cfg['tags'])

    logging.info(cfg)
    # datamodule = get_data(cfg)
    datamodule = SpectrogramsDataModule(config=cfg['dataset'])

    # 3. Build the model
    FLASH_MODELS = [
        "resnet18", 
        "resnet34", 
        "resnet50",
        "resnet101",
        "resnet152",
        "resnext50_32x4d",
        "resnext101_32x8d",
        "mobilenet_v2",
        "vgg11",
        "vgg13",
        "vgg16",
        "vgg19",
        "densenet121",
        "densenet169",
        "densenet161",
        "swav-imagenet",
    ]

    class_names = cfg['dataset']['classes_name']
    backbone_network = cfg['backbone_network']
    num_classes = len(class_names)

    if "efficientnet" in backbone_network:
        model = networks.EfficientNetPl(backbone_network, num_classes=num_classes)
    elif backbone_network in FLASH_MODELS:
        model = ImageClassifier(num_classes=num_classes, backbone=backbone_network, optimizer=torch.optim.Adam)
    else:
        raise NotImplementedError(f"Network: `{backbone_network}` not implemented.")


    # 4. Create the trainer. Run once on data
    model_folder  = os.path.join(current_folder, 'models-classifier')
    checkpoint_callback = ModelCheckpoint(model_folder, monitor='val_accuracy', verbose=True)
    trainer = flash.Trainer(
        logger=logger,
        gpus=cfg.get('gpus', 0),
        max_epochs=cfg.get('nb_epochs', 3),
        checkpoint_callback=checkpoint_callback,
    )
    logger.log_hyperparams(cfg)

    # 5. Fit the model
    logging.info("Training...")
    trainer.fit(model, datamodule=datamodule)
    
    logging.info("Testing...")
    logging.info("\tLoad best model...")
    saved_model =  os.listdir(model_folder)
    if len(saved_model) == 0:
        logging.info(f"No model found in {model_folder}")
        exit(1)
    elif len(saved_model) == 1:
        model_path = saved_model[0]
    else:
        logging.info(f"More than 1 model found in {model_folder}. We will load the last one using natsort.")
        import natsort
        model_path = natsort.natsorted(saved_model)[-1]
    model_path = os.path.join(model_folder , model_path)

    model.load_from_checkpoint(model_path)

    logging.info("\tInference...")
    predictions, targets = evaluate_model(model, datamodule.test_dataloader)
    logger.experiment.log({"confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=targets,
                        preds=predictions,
                        class_names=class_names)})
    
    cls_report = classification_report(targets, predictions, target_names=class_names, output_dict=True)
    test_log_metrics = {}
    for cls_name, metric in cls_report.items():
        if cls_name in class_names:
            for metric_name, metric_value in metric.items():
                test_log_metrics.update({ f"{metric_name}/{cls_name}": metric_value })
        else:
            test_log_metrics.update({ f"{cls_name}": metric})

    logger.log_metrics(test_log_metrics)
    logging.info(test_log_metrics)


if __name__ == "__main__":
    main()