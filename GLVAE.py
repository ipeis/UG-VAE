from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
from sklearn.manifold import TSNE
import os
import numpy as np
import matplotlib
matplotlib.use("Pdf")
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


class GLVAE(nn.Module):
    def __init__(self, dim_z, dim_Z):
        super(GLVAE, self).__init__()

        self.dim_z = dim_z
        self.dim_Z = dim_Z

        # Encoder
        self.fc1 = nn.Linear(784, 100)
        self.fc21 = nn.Linear(100, dim_z)
        self.fc22 = nn.Linear(100, dim_z)
        self.fc31 = nn.Linear(dim_z, dim_Z)
        self.fc32 = nn.Linear(dim_z, dim_Z)

        # Decoder
        self.fc3 = nn.Linear(dim_z+dim_Z, 100)
        self.fc4 = nn.Linear(100, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))

        mu = self.fc21(h1)
        std = torch.exp(0.5 * self.fc22(h1))

        return mu, std

    def global_encode(self, z_batch):
        mu = []
        prec = []
        for z in z_batch:
            mu.append(self.fc31(z))
            #varZ.append(self.fc32(z))
            prec.append(torch.exp(self.fc32(z))**-1)

        var = (torch.sum(torch.stack(prec), dim=0))**-1

        aux = torch.stack([torch.mul(p, mu[i]) for i, p in enumerate(prec)])
        aux = torch.sum(aux)

        mu = torch.mul(var, aux)

        return mu, var

    def reparameterize(self, mu, var):
        std = var**0.5
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, Z):
        aux = [torch.cat([z[i], Z]) for i in range(len(z))]
        aux = torch.stack(aux)
        h3 = F.relu(self.fc3(aux))

        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu_z, var_z = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu_z, var_z)
        mu_Z, var_Z = self.global_encode(z)
        Z = self.reparameterize(mu_Z, var_Z)

        return self.decode(z, Z), mu_z, var_z, mu_Z, var_Z


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, var, mu_g, var_g):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + torch.log(var) - mu.pow(2) - var)

    KLG = -0.5 * torch.sum(1 + torch.log(var_g) - mu_g.pow(2) - var_g)

    return BCE + KLD + KLG, BCE, KLD, KLG

