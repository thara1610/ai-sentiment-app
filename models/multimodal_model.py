import torch.nn as nn
import torch

class MultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2048 + 768, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 3)

    def forward(self, img_feat, text_feat):
        x = torch.cat((img_feat, text_feat), dim=1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)