import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, n_attr, z_channels=512, hid_dim=512, dropout=0.3):
        """
        n_attr     : nombre d'attributs à prédire (ex: 1 pour mono-attribut)
        z_channels : nb de canaux du latent (512 dans ton encodeur)
        hid_dim    : taille de la couche cachée fully-connected
        dropout    : probabilité de dropout (0.3 recommandé)
        """
        super().__init__()

        #2x2 -> 1x1  
        self.conv = nn.Sequential(
            nn.Conv2d(z_channels, z_channels, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(z_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # projection finale : 512 (1x1) -> hid_dim -> n_attr + dropout
        self.proj = nn.Sequential(
            nn.Linear(z_channels, hid_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hid_dim, n_attr),
        )

    def forward(self, z):
        """
        z : [B, 512, 2, 2] (sortie de l'encodeur)
        """
        B, C, H, W = z.shape
        assert (C, H, W) == (512, 2, 2), f"Discriminator attend [B, 512, 2, 2], reçu {z.shape}"

        h = self.conv(z)      # [B, 512, 1, 1]
        h = h.view(B, C)      # [B, 512]
        out = self.proj(h)    # [B, n_attr]
        return out
