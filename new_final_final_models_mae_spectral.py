# new_final_final_models_mae_spectral.py
# ---------------------------------------------------------
# Simple Autoencoder with 12 encoder + 12 decoder layers
# Encoder dims: 768 → 512 → 256 → 128
# Decoder dims: 128 → 256 → 512 → 768
# ---------------------------------------------------------

import torch
import torch.nn as nn


class SimpleAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim=768,
        hidden_dims_enc=[512, 256, 128],
        hidden_dims_dec=[256, 512, 768],
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

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == "__main__":
    model = SimpleAutoencoder()
    print(model)

    # create checkpoint with random weights
    torch.save(model.state_dict(), "Checkpoints/new_final_final_autoencoder.pth")
    print("✅ Saved initial random-weight checkpoint!")
