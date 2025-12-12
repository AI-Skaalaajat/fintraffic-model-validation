import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple
from torchvision import datasets
from tqdm.auto import tqdm
import model_setup
import utils

device = "cuda" if torch.cuda.is_available() else "cpu"
data_path = Path("software/data")
model_path = Path("software/models")
model_name = "ResNet-50.pth"
train_directory = data_path / "train"
test_directory = data_path / "test"
batch_size = 32
epochs = 5
num_workers = os.cpu_count()

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_function: torch.nn.Module,
               optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
    
    model.train()

    running_loss = 0
    running_accuracy = 0

    for batch, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.type(torch.float).to(device)
        labels = labels.type(torch.float).to(device)

        optimizer.zero_grad()

        outputs = model(inputs).squeeze()
        predicted_labels = torch.round(torch.sigmoid(outputs))

        loss = loss_function(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        running_accuracy += (predicted_labels == labels).sum().item() / len(labels)

    train_loss = running_loss / len(dataloader)
    train_accuracy = running_accuracy / len(dataloader)

    return train_loss, train_accuracy

def validation_step(model: torch.nn.Module,
                    dataloader: torch.utils.data.DataLoader,
                    loss_function: torch.nn.Module) -> Tuple[float, float]:
    
    model.eval()

    running_loss = 0
    running_accuracy = 0

    with torch.inference_mode():
        for batch, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.type(torch.float).to(device)
            labels = labels.type(torch.float).to(device)

            outputs = model(inputs).squeeze()
            predicted_labels = torch.round(torch.sigmoid(outputs))

            loss = loss_function(outputs, labels)
            running_loss += loss.item()

            running_accuracy += (predicted_labels == labels).sum().item() / len(labels)

    validation_loss = running_loss / len(dataloader)
    validation_accuracy = running_accuracy / len(dataloader)

    return validation_loss, validation_accuracy

def main():
    class_names = datasets.ImageFolder(train_directory).classes

    model, preprocess = model_setup.setup_resnet(layers=50,
                                                pretrained=True,
                                                class_names=class_names,
                                                device=device)
    
    dataset = datasets.ImageFolder(train_directory, transform=preprocess)
    train_set, validation_set = torch.utils.data.random_split(dataset, [0.8, 0.2])

    train_dataloader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=num_workers,
                                                   pin_memory=True)
    
    validation_dataloader = torch.utils.data.DataLoader(validation_set,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        num_workers=num_workers,
                                                        pin_memory=True)

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    results = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": []
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_accuracy = train_step(model=model,
                                                dataloader=train_dataloader,
                                                loss_function=loss_function,
                                                optimizer=optimizer)
        
        validation_loss, validation_accuracy = validation_step(model=model,
                                                               dataloader=validation_dataloader,
                                                               loss_function=loss_function)
        
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_accuracy: {train_accuracy:.2f}% | "
            f"validation_loss: {validation_loss:.4f} | "
            f"validation_accuracy: {validation_accuracy:.2f}%"
        )

        results["train_loss"].append(train_loss)
        results["train_accuracy"].append(train_accuracy)
        results["validation_loss"].append(validation_loss)
        results["validation_accuracy"].append(validation_accuracy)

    utils.plot_loss_curves(results)

    utils.save_model(model, model_path, model_name)

if __name__ == "__main__":
    main()