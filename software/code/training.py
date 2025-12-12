import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm.auto import tqdm
import data_setup
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

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_function: torch.nn.Module,
               optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
    
    model.train()

    train_loss = 0
    train_accuracy = 0

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device).type(torch.float)
        y = y.to(device).type(torch.float)

        optimizer.zero_grad()

        y_logits = model(X).squeeze()
        y_labels = torch.round(torch.sigmoid(y_logits))

        loss = loss_function(y_logits, y)
        train_loss += loss.item()
        train_accuracy += (y_labels == y).sum().item() / len(y_labels)

        
        loss.backward()
        optimizer.step()

    train_loss = train_loss / len(dataloader)
    train_accuracy = train_accuracy / len(dataloader)

    return train_loss, train_accuracy

def validation_step(model: torch.nn.Module,
                    dataloader: torch.utils.data.DataLoader,
                    loss_function: torch.nn.Module) -> Tuple[float, float]:
    
    model.eval()

    validation_loss = 0
    validation_accuracy = 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device).type(torch.float)
            y = y.to(device).type(torch.float)

            y_logits = model(X).squeeze()
            y_labels = torch.round(torch.sigmoid(y_logits))

            loss = loss_function(y_logits, y)
            validation_loss += loss.item()

            validation_accuracy += (y_labels == y).sum().item() / len(y_labels)

    validation_loss = validation_loss / len(dataloader)
    validation_accuracy = validation_accuracy / len(dataloader)

    return validation_loss, validation_accuracy

def main():
    class_names = data_setup.get_class_names(train_directory)

    model, preprocess = model_setup.setup_resnet(layers=50,
                                                pretrained=True,
                                                class_names=class_names,
                                                device=device)

    train_dataloader = data_setup.create_dataloader(directory=train_directory,
                                                    transform=preprocess,
                                                    batch_size=batch_size,
                                                    shuffle=True)

    validation_dataloader = data_setup.create_dataloader(directory=test_directory,
                                                         transform=preprocess,
                                                         batch_size=batch_size,
                                                         shuffle=False)

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

    utils.save_model(model, model_path, model_name)

if __name__ == "__main__":
    main()