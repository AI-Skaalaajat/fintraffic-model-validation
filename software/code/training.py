import torch
import torch.nn as nn
from pathlib import Path
import data_setup
import model_setup
import training_loop

device = "cuda" if torch.cuda.is_available() else "cpu"
data_path = Path("software/data")
train_directory = data_path / "train"
test_directory = data_path / "test"
batch_size = 32
epochs = 5
class_names = data_setup.get_class_names(train_directory)

model, preprocess = model_setup.setup_resnet(layers=50,
                                             pretrained=True,
                                             class_names=class_names,
                                             device=device)

train_dataloader = data_setup.create_dataloader(directory=train_directory,
                                                transform=preprocess,
                                                batch_size=batch_size,
                                                shuffle=True)

test_dataloader = data_setup.create_dataloader(directory=test_directory,
                                               transform=preprocess,
                                               batch_size=batch_size,
                                               shuffle=False)

loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

results = training_loop.train(model=model,
                              train_dataloader=train_dataloader,
                              test_dataloader=test_dataloader,
                              loss_function=loss_function,
                              optimizer=optimizer,
                              epochs=epochs,
                              device=device)