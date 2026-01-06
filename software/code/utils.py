import torch
from pathlib import Path
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, List
import configparser

def read_config():
    config = configparser.ConfigParser()
    config.read('software/code/config.ini')

    train_directory = config.get('Directory', 'train_directory')
    test_directory = config.get('Directory', 'test_directory')
    model_directory = config.get('Directory', 'model_directory')

    model = config.get('Model', 'model')
    pretrained = config.getboolean('Model', 'pretrained')
    size = config.get('Model', 'size')
    resnet_layers = config.getint('Model', 'resnet_layers')
    swin_transformer_version = config.getint('Model', 'swin_transformer_version')
    model_file_name = config.get('Model', 'model_file_name')

    batch_size = config.getint('Training', 'batch_size')
    epochs = config.getint('Training', 'epochs')
    learning_rate = config.getfloat('Training', 'learning_rate')

    config_values = {
        'train_directory': train_directory,
        'test_directory': test_directory,
        'model_directory': model_directory,
        'model': model,
        'pretrained': pretrained,
        'size': size,
        'resnet_layers': resnet_layers,
        'swin_transformer_version': swin_transformer_version,
        'model_file_name': model_file_name,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate
    }

    return config_values

def save_model(model: nn.Module,
               directory: Path,
               model_name: str):
    
    model_path = directory / model_name
    torch.save(obj=model.state_dict(),
               f=model_path)
    
def plot_loss_curves(results: Dict[str, List[float]]):
    train_loss = results['train_loss']
    validation_loss = results['validation_loss']
    epochs = range(len(results['train_loss']))

    plt.figure(figsize=(15, 7))
    plt.plot(epochs, train_loss, label='train_loss')
    plt.plot(epochs, validation_loss, label='validation_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()