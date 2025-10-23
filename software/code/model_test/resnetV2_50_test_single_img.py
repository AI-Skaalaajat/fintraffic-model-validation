import torch
from torchvision import transforms
from PIL import Image
import timm
import os

# configuration
MODEL_NAME = "resnetv2_50x1_bit.goog_in21k_ft_in1k"

# --------------------------different models selection--------------------------
# PIC_SIZE = 128
# MODEL_PATH = "../../models/resnetV2_final_128.pth"
# ----------------------------------------------------
PIC_SIZE = 224
# MODEL_PATH = "../../models/resnetV2_final_224.pth" # full fine-tuning model
# MODEL_PATH = "../../models/resnetV2_final_fine-tune_0.98.pth" #Partial fine-tuning model
MODEL_PATH = "../../models/resnetV2_final_fine_tune_weight_0.97.pth"  # Partial fine-tuning model(Model with adjusted weights)
# MODEL_PATH = "../../models/resnetV2.pth"  # same as resnetV2_final_fine_tune_weight_0.97.pth, but no rejection threshold saved
# --------------------------different models selection--------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DFAULT_THRESHOLD = 0.85  # default threshold if not found in checkpoint
CLASS_NAMES = ["dry", "wet"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# data transform
transform = transforms.Compose([
    transforms.Resize((PIC_SIZE, PIC_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# load model
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=2)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

# load threshold
threshold = checkpoint.get("optimal_threshold", DFAULT_THRESHOLD)
print(f"Using rejection threshold: {threshold:.3f}")


def predict_image(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"[Warning] Failed to open image: {img_path}. Error: {e}")
        return "error", 0.0

    try:
        tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            max_prob, pred = torch.max(probs, 1)
            max_prob = max_prob.item()
            pred = pred.item()
    except Exception as e:
        print(f"[Warning] Inference failed on: {img_path}. Error: {e}")
        return "error", 0.0

    if max_prob < threshold:
        return "rejected", max_prob
    return CLASS_NAMES[pred], max_prob


if __name__ == "__main__":
    test_dir = "../../single_image_test"  # store your model_test images here
    for fname in os.listdir(test_dir):
        fpath = os.path.join(test_dir, fname)
        label, prob = predict_image(fpath)
        print(f"{fname}: {label} (confidence={prob:.3f})")
