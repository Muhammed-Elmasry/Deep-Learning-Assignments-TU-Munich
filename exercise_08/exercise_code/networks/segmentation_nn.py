"""SegmentationNN"""
import torch
import torch.nn as nn

class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn
from torchvision.models import mobilenet_v2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mobnet = mobilenet_v2().to(device)
encoder = mobnet.features

class SegmentationNN(nn.Module):
    def __init__(self, num_classes=23, dropout_prob=0.5):
        super(SegmentationNN, self).__init__()
        self.encoder = encoder

        # The last channel dimension of the encoder's output
        encoder_out_channels = self.encoder[-1].out_channels

        # # Adding dropout layers
        # self.dropout1 = nn.Dropout2d(dropout_prob)
        # self.dropout2 = nn.Dropout2d(dropout_prob)

        # Adding upsampling layers
        self.upsample = nn.Sequential(
            nn.Conv2d(encoder_out_channels, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            # self.dropout1,
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # self.dropout2,
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, num_classes, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(240, 240), mode='bilinear', align_corners=False),
        )
    
    def forward(self, x):
        # Pass input through the encoder
        x = self.encoder(x)
        # Upsample the features to the target size
        x = self.upsample(x)
        return x


    # @property
    # def is_cuda(self):
    #     """
    #     Check if model parameters are allocated on the GPU.
    #     """
    #     return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(nn.Module):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()

if __name__ == "__main__":
    from torchinfo import summary
    summary(SegmentationNN(), (1, 3, 240, 240), device="cpu")