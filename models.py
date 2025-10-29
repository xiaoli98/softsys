import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ConfigurableCNN(nn.Module):
    def __init__(
        self,
        num_classes,
        pooling_size,
        in_channels=1,
        depth=3,
        kernel_size=3,
        stride=1,
        padding='same',
        latent_dim=128,
        use_contrastive=False,
        projection_dim=128):
        super().__init__()
        self.use_contrastive = use_contrastive
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
        # to capture every high level feature
        layers.append(nn.AdaptiveMaxPool2d(output_size=pooling_size))
        self.feature_extractor = nn.Sequential(*layers)
        
        # calculate the flattened size
        # with torch.no_grad():
        #     dummy_input = torch.randn(1, 1, *img_size)
        #     dummy_output = self.feature_extractor(dummy_input)
        #     flattened_size = dummy_output.view(1, -1).shape[1]

        flattened_size = current_channels // 2 * pooling_size[0] * pooling_size[1]
        
        self.classifier_backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        if num_classes == 2:
            self.classifier_head = nn.Linear(latent_dim, 1)
        else:
            self.classifier_head = nn.Linear(latent_dim, num_classes)

        # Conditionally create the projection head
        if self.use_contrastive:
            self.projection_head = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(inplace=True),
                nn.Linear(latent_dim, projection_dim)
            )

    def forward(self, x):
        x = self.feature_extractor(x)
        features = self.classifier_backbone(x)
        logits = self.classifier_head(features)
        logits = F.sigmoid(logits)
        
        # Return projections only if the head exists and is requested
        if self.use_contrastive:
            projections = self.projection_head(features)
            return logits, F.normalize(projections, dim=1)
        
        return logits

class VisionTransformer(nn.Module):
    def __init__(self, num_classes, latent_dim=768, pretrained=True, use_contrastive=False, projection_dim=128): # <-- Add flag
        super().__init__()
        self.use_contrastive = use_contrastive
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = vit_b_16(weights=weights)

        # Modify the patch embedding layer to accept 1 channel instead of 3
        original_conv_proj = self.vit.conv_proj
        self.vit.conv_proj = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv_proj.out_channels,
            kernel_size=original_conv_proj.kernel_size,
            stride=original_conv_proj.stride,
            padding=original_conv_proj.padding,
            bias=original_conv_proj.bias is not None
        )
        # Note: The weights for this new conv_proj layer will be randomly initialized.

        original_in_features = self.vit.heads.head.in_features
        if num_classes == 2:
            self.vit.heads.head = nn.Linear(original_in_features, 1)
        else:
            self.vit.heads.head = nn.Linear(original_in_features, num_classes)

        # Conditionally create the projection head
        if self.use_contrastive:
            self.projection_head = nn.Sequential(
                nn.Linear(original_in_features, original_in_features),
                nn.ReLU(inplace=True),
                nn.Linear(original_in_features, projection_dim)
            )
        
    def forward(self, x):
        features = self.vit._process_input(x)
        n = features.shape[0]
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        features = torch.cat([batch_class_token, features], dim=1)
        features = self.vit.encoder(features)
        features = features[:, 0]

        logits = self.vit.heads(features)
        logits = F.sigmoid(logits)
        # Return projections only if the head exists and is requested
        if self.use_contrastive:
            projections = self.projection_head(features)
            return logits, F.normalize(projections, dim=1)

        return logits