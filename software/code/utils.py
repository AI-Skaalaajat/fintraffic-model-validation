import torch
from pathlib import Path
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, List

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