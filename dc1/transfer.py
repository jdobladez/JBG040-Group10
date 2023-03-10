import torch
import torch.nn as nn
import torchvision.models as models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class PreNet(nn.Module):
    def __init__(self, n_classes: int, pretrained_network: str) -> None:
        super(PreNet, self).__init__()
        if pretrained_network == 'resnet18':
            self.pretrained_model = models.resnet18('ResNet18_Weights.IMAGENET1K_V1')
        # elif pretrained_network == 'vgg16':
        #     self.pretrained_model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
        # Loading pre-trained model

        # Freezing all layers in the pre-trained model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.pretrained_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)


        # Replacing the last layer in the pre-trained model to match our number of classes
        num_ftrs = self.pretrained_model.fc.in_features
        self.pretrained_model.fc = nn.Linear(num_ftrs, n_classes)

    # Defining the forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Using the pre-trained model to extract features
        x = self.pretrained_model(x)
        return x