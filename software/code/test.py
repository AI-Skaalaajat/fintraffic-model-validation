import torch
from torchvision import datasets
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import model_setup
from utils import read_config

config = read_config()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
test_directory = Path(config['test_directory'])
model_directory = Path(config['model_directory'])
model_name = config['model_file_name']

def main():
    class_names = datasets.ImageFolder(test_directory).classes

    model, preprocess = model_setup.setup(model=config['model'],
                                          pretrained=False,
                                          class_names=class_names,
                                          device=device,
                                          size=config['size'],
                                          resnet_layers=config['resnet_layers'],
                                          swin_transformer_version=config['swin_transformer_version'])

    model.load_state_dict(torch.load(f=model_directory / model_name))

    test_set = datasets.ImageFolder(test_directory, transform=preprocess)
    test_dataloader = torch.utils.data.DataLoader(test_set)

    model.eval()

    label_indexes = []
    for i in range(len(class_names)):
        label_indexes.append(i)

    true_labels = []
    predicted_labels = []

    with torch.inference_mode():
        for input, label in tqdm(test_dataloader):
            input, label = input.to(device), label.to(device)

            output = model(input)
            predicted_label = torch.argmax(torch.softmax(output, dim=1), dim=1)
            
            true_labels.append(label.item())
            predicted_labels.append(predicted_label.item())

        total_accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, labels=label_indexes, average=None)
        recall = recall_score(true_labels, predicted_labels, labels=label_indexes, average=None)
        f1 = f1_score(true_labels, predicted_labels, labels=label_indexes, average=None)

        print(f'Total accuracy: {total_accuracy}\n')

        for i in range(len(class_names)):
            class_name = class_names[i]

            print(f'Metrics for "{class_name}" class:')
            print(f'Precision: {precision[i]}')
            print(f'Recall: {recall[i]}')
            print(f'F1 Score: {f1[i]}\n')

        matrix = confusion_matrix(true_labels, predicted_labels)
        disp = ConfusionMatrixDisplay(matrix, display_labels=class_names)
        disp.plot()
        plt.show()

if __name__ == '__main__':
    main()