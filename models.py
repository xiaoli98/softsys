import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ConfigurableCNN(nn.Module):
    def __init__(
        self,
        num_classes,
        img_size,
        in_channels=3,
        depth=3,
        kernel_size=3,
        stride=1,
        padding='same',
        latent_dim=128):
        super().__init__()
        layers = []
        
        current_channels = 32
        for i in range(depth):
            layers.append(
                nn.Conv2d(in_channels, current_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            )
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(current_channels))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = current_channels
            current_channels *= 2
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # calculate the flattened size
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size, img_size)
            dummy_output = self.feature_extractor(dummy_input)
            flattened_size = dummy_output.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(latent_dim, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, num_classes, latent_dim=768, pretrained=True):
        super().__init__()
        # Note: 'latent_dim' for ViT is typically its embedding dimension, which is fixed for pretrained models.
        # We will use the standard ViT-B/16 model. `latent_dim` here is more of a placeholder.
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = vit_b_16(weights=weights)

        # Replace the final classification head
        original_in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(original_in_features, num_classes)
        
    def forward(self, x):
        return self.vit(x)