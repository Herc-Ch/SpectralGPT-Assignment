# new_final_final_models_mae_spectral.py
# ---------------------------------------------------------
# Two architectures:
#  - SimpleAutoencoder (fully connected)
#  - CNN_Autoencoder (convolutional)
# Both share the same forward() logic for consistency.
# ---------------------------------------------------------

import torch
import torch.nn as nn


class BaseAutoencoder(nn.Module):
    encoder: nn.Module
    decoder: nn.Module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class SimpleAutoencoder(BaseAutoencoder):
    def __init__(
        self,
        input_dim=768,
        hidden_dims_enc=[640, 512, 384, 256, 192, 128],
        hidden_dims_dec=[192, 256, 384, 512, 640, 768],
    ):
        super().__init__()

        # ----- Encoder -----
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims_enc:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.encoder = nn.Sequential(*layers)

        # ----- Decoder -----
        layers = []
        in_dim = hidden_dims_enc[-1]
        for h_dim in hidden_dims_dec:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.decoder = nn.Sequential(*layers)

    # def forward(self, x):
    #     encoded = self.encoder(x)
    #     decoded = self.decoder(encoded)
    #     return decoded


class CNN_Autoencoder(BaseAutoencoder):
    def __init__(self):
        super().__init__()

        # ---- Encoder (12 layers total) ----
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 1
            nn.ReLU(),  # 2
            nn.Conv2d(32, 64, 3, padding=1),  # 3
            nn.ReLU(),  # 4
            nn.MaxPool2d(2, 2),  # downsample 32x24 → 16x12
            nn.Conv2d(64, 128, 3, padding=1),  # 5
            nn.ReLU(),  # 6
            nn.Conv2d(128, 256, 3, padding=1),  # 7
            nn.ReLU(),  # 8
            nn.MaxPool2d(2, 2),  # downsample again
            nn.Conv2d(256, 512, 3, padding=1),  # 9
            nn.ReLU(),  # 10
            nn.Conv2d(512, 128, 3, padding=1),  # 11 (bottleneck start)
            nn.ReLU(),  # 12
        )

        # ---- Decoder (12 layers total) ----
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 512, 3, padding=1),  # 1
            nn.ReLU(),  # 2
            nn.ConvTranspose2d(512, 256, 3, padding=1),  # 3
            nn.ReLU(),  # 4
            nn.Upsample(scale_factor=2, mode="nearest"),  # upsample
            nn.ConvTranspose2d(256, 128, 3, padding=1),  # 5
            nn.ReLU(),  # 6
            nn.ConvTranspose2d(128, 64, 3, padding=1),  # 7
            nn.ReLU(),  # 8
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ConvTranspose2d(64, 32, 3, padding=1),  # 9
            nn.ReLU(),  # 10
            nn.ConvTranspose2d(32, 1, 3, padding=1),  # 11
            nn.Sigmoid(),  # 12 (output)
        )

    # def forward(self, x):
    #     encoded = self.encoder(x)
    #     decoded = self.decoder(encoded)
    #     return decoded
