import torch
from torchvision import datasets
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import model_setup

device = "cuda" if torch.cuda.is_available() else "cpu"
data_path = Path("software/data")
model_path = Path("software/models")
test_directory = data_path / "test"
model_name = "ResNet-50.pth"

def main():
    class_names = datasets.ImageFolder(test_directory).classes

    model, preprocess = model_setup.setup_resnet(layers=50,
                                                pretrained=False,
                                                class_names=class_names,
                                                device=device)

    model.load_state_dict(torch.load(f=model_path / model_name))

    test_set = datasets.ImageFolder(test_directory, transform=preprocess)
    test_dataloader = torch.utils.data.DataLoader(test_set)

    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.inference_mode():
        for input, label in tqdm(test_dataloader):
            input, label = input.to(device), label.to(device)

            output = model(input)
            predicted_label = torch.argmax(torch.softmax(output, dim=1), dim=1)
            
            true_labels.append(label.item())
            predicted_labels.append(predicted_label.cpu())

        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 score: {f1}")

        confusion_matrix = confusion_matrix(true_labels, predicted_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix, display_labels=class_names)
        disp.plot()
        plt.show()

if __name__ == "__main__":
    main()