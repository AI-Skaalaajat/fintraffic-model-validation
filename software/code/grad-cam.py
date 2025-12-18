import os
from pathlib import Path
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import model_setup

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = Path("software/models")
data_path = Path("software/data/test")
output_path = Path("software/outputs/grad-cam")

image_path = data_path / "dry/C03676_dry_11_2025-06-23T10_20_43Z.jpg"
model_name = "ConvNeXt-Base_Pretrained.pth"
model_type = "convnext"

class_names = datasets.ImageFolder(data_path).classes
model, preprocess = model_setup.setup_convnext(size="base",
                                               pretrained=False,
                                               class_names=class_names,
                                               device=device)

# Get the last convolutional layer of the model
def get_target_layer(model, model_type):
    if model_type == "convnext":
        return model.features[-1]
    
    if model_type == "resnet":
        return model.layer4
    
    if model_type == "swintransformer":
        return model.features[-1][-1].norm1

def create_cam_image(model, image, image_tensor):
    target_layers = [get_target_layer(model, model_type)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=image_tensor)
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(image, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    return cam_image

def display_cam_image(cam_image, title):
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.imread(image_path)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].imshow(original_image)
    ax[0].axis("off")
    ax[1].imshow(cam_image)
    ax[1].axis("off")

    fig.tight_layout()
    fig.suptitle(title)

    save_figure(plt, "output")
    plt.show()

def save_figure(figure, file_name):
    i = 0
    while os.path.exists(f"{output_path}/{file_name}{i}.jpg"):
        i += 1
    figure.savefig(f"{output_path}/{file_name}{i}.jpg", bbox_inches="tight", pad_inches=0.1)

def main():
    print(model)
    model.load_state_dict(torch.load(f=model_path / model_name))
    
    image = cv2.imread(image_path, 1)[:, :, ::-1]
    image = np.float32(image) / 255
    image_tensor = preprocess_image(image,
                                    mean=preprocess.mean,
                                    std=preprocess.std).to(device)
    
    cam_image = create_cam_image(model, image, image_tensor)
    display_cam_image(cam_image, "test")

if __name__ == "__main__":
    main()