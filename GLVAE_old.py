from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from sklearn.manifold import TSNE
import os
import numpy as np
import matplotlib
matplotlib.use("Pdf")
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from torch.nn.modules.flatten import Flatten

class GLVAE(nn.Module):
    def __init__(self, channels, dim_z, dim_Z, arch='beta_vae'):
        super(GLVAE, self).__init__()

        self.dim_z = dim_z
        self.dim_Z = dim_Z

        # Architecture from beta_vae
        if arch=='beta_vae':
            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(channels, 32, 4, 2, 1),  # B,  32, 32, 32
                nn.ReLU(True),
                nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
                nn.ReLU(True),
                nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  8,  8
                nn.ReLU(True),
                nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  4,  4
                nn.ReLU(True),
                nn.Conv2d(64, 256, 4, 1),  # B, 256,  1,  1
                nn.ReLU(True),
                View((-1, 256 * 1 * 1)),  # B, 256
                nn.Linear(256, dim_z * 2),  # B, z_dim*2
            )

            self.decoder = nn.Sequential(
                nn.Linear(dim_z+dim_Z, 256),  # B, 256
                View((-1, 256, 1, 1)),  # B, 256,  1,  1
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 64, 4),  # B,  64,  4,  4
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 64, 4, 2, 1),  # B,  64,  8,  8
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  32, 16, 16
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
                nn.ReLU(True),
                nn.ConvTranspose2d(32, channels, 4, 2, 1),  # B, nc, 64, 64
                nn.Sigmoid()
            )

        # Original architecture in Kingma's VAE
        elif arch=='k_vae':
            self.encoder = nn.Sequential(
                View((-1, channels*784)),
                nn.Linear(channels*784, 400),
                nn.ReLU(),
                nn.Linear(400, dim_z*2)
            )
            self.decoder = nn.Sequential(
                nn.Linear(dim_z+dim_Z, 400),
                nn.ReLU(),
                nn.Linear(400, channels*784),
                nn.Sigmoid(),
                View((-1, channels, 28, 28))
            )

        self.global_encoder = nn.Sequential(
            nn.Linear(dim_z, 256),
            nn.ReLU(True),
            nn.Linear(256, dim_Z * 2),
        )

    def _encode(self, x):
        theta = self.encoder(x)
        mu = theta[:, :self.dim_z]
        std = torch.exp(0.5 * theta[:, self.dim_z:] )

        return mu, std

    def _global_encode(self, z_batch):
        mu = []
        prec = []
        theta = self.global_encoder(z_batch)
        mu = theta[:, :self.dim_Z]
        prec = torch.exp(theta[:, self.dim_Z:]) ** -1

        var = (torch.sum(prec, dim=0))**-1
        aux = torch.sum(torch.mul(prec, mu))
        mu = torch.mul(var, aux)

        return mu, var

    def reparameterize(self, mu, var):
        std = var**0.5
        eps = torch.randn_like(std)
        return mu + eps*std

    def _decode(self, z, Z):
        z_gl = [torch.cat([z[i], Z]) for i in range(len(z))]
        z_gl = torch.stack(z_gl)

        return self.decoder(z_gl)

    def forward(self, x):
        mu_z, var_z = self._encode(x)
        z = self.reparameterize(mu_z, var_z)
        mu_Z, var_Z = self._global_encode(z)
        Z = self.reparameterize(mu_Z, var_Z)

        return self._decode(z, Z), mu_z, var_z, mu_Z, var_Z

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, var, mu_g, var_g, beta_l=1.0, beta_g=1.0, distribution='bernoulli'):

    #BCE = F.binary_cross_entropy(recon_x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]),
    #                             x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]),
    #                             reduction='sum')
    if distribution=='bernoulli':
        recogn = F.binary_cross_entropy(recon_x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]),
                                 x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]),
                                        reduction='sum')
    elif distribution=='gaussian':
        recogn = F.mse_loss(recon_x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]),
                            x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + torch.log(var) - mu.pow(2) - var)

    KLG = -0.5 * torch.sum(1 + torch.log(var_g) - mu_g.pow(2) - var_g)

    return recogn + beta_l*KLD + beta_g*KLG, recogn, beta_l*KLD, beta_g*KLG

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)