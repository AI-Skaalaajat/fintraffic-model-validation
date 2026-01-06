import os
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import datasets
import engine
import model_setup
import utils
from utils import read_config

config = read_config()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_directory = Path(config['train_directory'])
model_directory = Path(config['model_directory'])
num_workers = os.cpu_count()
class_names = datasets.ImageFolder(train_directory).classes

def main():
    model, preprocess = model_setup.setup_resnet(layers=50,
                                                 pretrained=True,
                                                 class_names=class_names,
                                                 device=device)

    dataset = datasets.ImageFolder(train_directory, transform=preprocess)
    train_set, validation_set = torch.utils.data.random_split(dataset, [0.8, 0.2])

    train_dataloader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=config['batch_size'],
                                                   shuffle=True,
                                                   num_workers=num_workers,
                                                   pin_memory=True)
    
    validation_dataloader = torch.utils.data.DataLoader(validation_set,
                                                        batch_size=config['batch_size'],
                                                        shuffle=False,
                                                        num_workers=num_workers,
                                                        pin_memory=True)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    results = engine.train(model=model,
                           train_dataloader=train_dataloader,
                           validation_dataloader=validation_dataloader,
                           loss_function=loss_function,
                           optimizer=optimizer,
                           epochs=config['epochs'],
                           device=device)

    utils.plot_loss_curves(results)
    utils.save_model(model, model_directory, config['output_file_name'])

if __name__ == '__main__':
    main()