import torch
from torch import nn
from torch.nn import functional as F


class CNN(nn.Module):
    def __init__(self, in_channels, dropout=0.25, norm_type=5):
        super(CNN, self).__init__()

        self.norm_type = norm_type

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            nn.MaxPool2d(kernel_size=(1, 4)),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            nn.MaxPool2d(kernel_size=(1, 4)),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            nn.MaxPool2d(kernel_size=(1, 2)))

        self.fc = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        # Dimension of x along temporal axis
        x_dim1 = x.shape[2]

        x = self.conv_layers(x)
        x = x.permute(0, 2, 1, -1).squeeze()
        x = torch.sigmoid(self.fc(x))
        x = F.lp_pool2d(x, norm_type=self.norm_type, kernel_size=(x_dim1, 1))
        x = torch.mul(x, (1 / x_dim1) ** (1 / self.norm_type))
        x = x.view(-1, 1)

        return x
