from torchvision import datasets, transforms
import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from torch import nn, cuda, autocast, torch
import timm
import shutil

MODEL_NAME = "resnetv2_50x1_bit.goog_in21k_ft_in1k"

# --------------------------different models selection--------------------------
# PIC_SIZE = 128
# MODEL_PATH = "../../models/resnetV2_final_128.pth"
# ----------------------------------------------------
PIC_SIZE = 224
# MODEL_PATH = "../../models/resnetV2_final_224.pth" # full fine-tuning model
# MODEL_PATH = "../../models/resnetV2_final_fine-tune_0.98.pth" #Partial fine-tuning model
MODEL_PATH = "../../models/resnetV2_final_fine_tune_weight_0.97.pth"  # Partial fine-tuning model(Model with adjusted weights)
# MODEL_PATH = "../../models/resnetV2.pth"  # same as resnetV2_final_fine_tune_weight_0.97.pth, but no rejection threshold saved, so we can modify default threshold to test
# --------------------------different models selection--------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "../../dataset"
REJECTED_DIR = "../../rejected_images"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DFAULT_THRESHOLD = 0.95  # default threshold if not found in checkpoint


def evaluate_with_rejection(model, dataloader, threshold, save_rejected=False, phase='test'):
    global rejected_path
    model.eval()
    running_corrects = 0
    running_rejected = 0
    all_preds = []
    all_labels = []
    all_probs = []
    rejected_indices = []

    if save_rejected:
        rejected_path = os.path.join(REJECTED_DIR, phase)
        if os.path.exists(rejected_path):
            shutil.rmtree(rejected_path)
        os.makedirs(rejected_path, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            with autocast("cuda" if cuda.is_available() else "cpu"):
                outputs = model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                max_probs, preds = torch.max(probs, 1)

            base_idx = len(all_labels)
            for i in range(len(preds)):
                global_idx = base_idx + i

                if max_probs[i] < threshold:
                    all_preds.append(2)  # "unknown/rejected"
                    running_rejected += 1
                    rejected_indices.append(global_idx)

                    # save rejected images
                    if save_rejected and global_idx < len(dataloader.dataset):
                        img_path, _ = dataloader.dataset.imgs[global_idx]
                        img_name = os.path.basename(img_path)
                        label_dir = os.path.basename(os.path.dirname(img_path))
                        save_subdir = os.path.join(rejected_path, label_dir)
                        os.makedirs(save_subdir, exist_ok=True)
                        save_path = os.path.join(save_subdir, f"rejected_{img_name}")
                        shutil.copy2(img_path, save_path)
                else:
                    all_preds.append(preds[i].item())
                    if preds[i] == labels[i]:
                        running_corrects += 1

                all_labels.append(labels[i].item())
                all_probs.append(max_probs[i].item())

    total_samples = len(dataloader.dataset)
    accepted_samples = total_samples - running_rejected

    # calculate accuracy on accepted samples
    if accepted_samples > 0:
        accuracy_on_accepted = running_corrects / accepted_samples
    else:
        accuracy_on_accepted = 0

    overall_accuracy = running_corrects / total_samples
    rejection_rate = running_rejected / total_samples

    # calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])

    return {
        'accuracy_on_accepted': accuracy_on_accepted,
        'overall_accuracy': overall_accuracy,
        'rejection_rate': rejection_rate,
        'total_samples': total_samples,
        'accepted_samples': accepted_samples,
        'rejected_samples': running_rejected,
        'confusion_matrix': cm,
        'all_preds': all_preds,
        'all_labels': all_labels,
        'all_probs': all_probs,
        'rejected_indices': rejected_indices
    }

# evaluate model function
def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    cm = confusion_matrix(all_labels, all_preds)

    return epoch_loss, epoch_acc, precision, recall, f1, cm, all_preds, all_labels


test_transform = transforms.Compose([
    transforms.Resize((PIC_SIZE, PIC_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# load test dataset
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=2)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

# load model and threshold
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    class_names = checkpoint.get('class_names', test_dataset.classes)
    threshold = checkpoint.get('optimal_threshold', DFAULT_THRESHOLD)  # 默认值0.95
else:
    model.load_state_dict(checkpoint)
    class_names = test_dataset.classes
    threshold = DFAULT_THRESHOLD
print(f"Using rejection threshold: {threshold:.4f}")

model = model.to(DEVICE)
model.eval()

criterion = nn.CrossEntropyLoss()

loss, acc, precision, recall, f1, cm, preds, labels = evaluate_model(
    model, test_loader, criterion
)

y_pred = np.asarray(preds).astype(int)
y_true = np.asarray(labels).astype(int)

#
target_names = class_names

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=target_names, digits=2))

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

print(f"Test (no rejection) - Loss: {loss:.4f}, Acc: {acc:.4f}, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")
results = evaluate_with_rejection(model, test_loader, threshold, save_rejected=True, phase='test')
print(f"Test (with rejection, threshold={threshold:.2f})")
print(f"  Accuracy on accepted: {results['accuracy_on_accepted']:.4f}")
print(f"  Overall accuracy: {results['overall_accuracy']:.4f}")
print(f"  Rejection rate: {results['rejection_rate']:.4f}")
