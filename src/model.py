import timm
import torch
import torch.nn as nn

def load_dino_vits16():
    # Load pretrained DINO ViT-S/16 backbone
    model = timm.create_model('vit_small_patch16_224_dino', pretrained=True)
    
    # Remove the classification head
    model.head = nn.Identity()
    
    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False
    
    return model

class DINOClassifier(nn.Module):
    def __init__(self, backbone, num_classes=100):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
