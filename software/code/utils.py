import torch
from pathlib import Path
import torch.nn as nn

def save_model(model: nn.Module,
               directory: Path,
               model_name: str):
    
    model_path = directory / model_name
    torch.save(obj=model.state_dict(),
               f=model_path)