import torch
import torch.nn as nn
import torchvision.models as models

class PreNet(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(PreNet, self).__init__()

        # Loading pre-trained model
        self.pretrained_model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')

        # Freezing all layers in the pre-trained model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        # Replacing the last layer in the pre-trained model to match our number of classes
        num_ftrs = self.pretrained_model.fc.in_features
        self.pretrained_model.fc = nn.Linear(num_ftrs, n_classes)

    # Defining the forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Using the pre-trained model to extract features
        x = self.pretrained_model(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x