import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

model = models.resnet50(pretrained=True)
model.fc = nn.Identity()

def get_image_features(path):
    img = Image.open(path).convert('RGB')
    img = transform(img).unsqueeze(0)
    return model(img)