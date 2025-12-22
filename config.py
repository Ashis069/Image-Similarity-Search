import torch 
from torchvision import transforms


# This centralizes the device logic
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Project initialized using: {device}")


# Global Transform
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),                 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

