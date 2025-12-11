import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_function: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    
    model.train()

    train_loss = 0
    train_accuracy = 0

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device).type(torch.float)
        y = y.to(device).type(torch.float)

        y_logits = model(X).squeeze()
        y_labels = torch.round(torch.sigmoid(y_logits))

        loss = loss_function(y_logits, y)
        train_loss += loss.item()
        train_accuracy += (y_labels == y).sum().item() / len(y_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = train_loss / len(dataloader)
    train_accuracy = train_accuracy / len(dataloader)

    return train_loss, train_accuracy

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_function: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    
    model.eval()

    test_loss = 0
    test_accuracy = 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device).type(torch.float)
            y = y.to(device).type(torch.float)

            y_logits = model(X).squeeze()
            y_labels = torch.round(torch.sigmoid(y_logits))

            loss = loss_function(y_logits, y)
            test_loss += loss.item()

            test_accuracy += (y_labels == y).sum().item() / len(y_labels)

    test_loss = test_loss / len(dataloader)
    test_accuracy = test_accuracy / len(dataloader)

    return test_loss, test_accuracy

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_function: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    
    results = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": []
    }

    for epoch in tqdm(range(epochs)):

        train_loss, train_accuracy = train_step(model=model,
                                                dataloader=train_dataloader,
                                                loss_function=loss_function,
                                                optimizer=optimizer,
                                                device=device)
        
        test_loss, test_accuracy = test_step(model=model,
                                             dataloader=test_dataloader,
                                             loss_function=loss_function,
                                             device=device)
        
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_accuracy: {train_accuracy:.2f}% | "
            f"test_loss: {test_loss:.4f} | "
            f"test_accuracy: {test_accuracy:.2f}%"
        )

        results["train_loss"].append(train_loss)
        results["train_accuracy"].append(train_accuracy)
        results["test_loss"].append(test_loss)
        results["test_accuracy"].append(test_accuracy)

    return results