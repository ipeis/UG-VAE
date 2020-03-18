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
from torch.nn.modules.flatten import Flatten

class GLVAE(nn.Module):
    def __init__(self, dim_z, dim_Z):
        super(GLVAE, self).__init__()

        self.dim_z = dim_z
        self.dim_Z = dim_Z

        # Encoder
        self.conv11 = torch.nn.Conv2d(in_channels=3, out_channels=28, kernel_size=4, stride=2, padding=1)
        self.conv12 = torch.nn.Conv2d(in_channels=28, out_channels=28, kernel_size=4, stride=2, padding=1)
        self.conv13 = torch.nn.Conv2d(in_channels=28, out_channels=28, kernel_size=4, stride=2, padding=1)
        self.conv14 = nn.Sequential(torch.nn.Conv2d(in_channels=28, out_channels=28, kernel_size=4, stride=2, padding=1),
                                   Flatten())
        self.fc11 = nn.Linear(28, 256)
        self.fc12 = nn.Linear(256, 256)
        self.fc13 = nn.Linear(256, 2*dim_z)

        # Global encoder
        #self.fc21 = nn.Linear(dim_z, 256)
        self.fc21 = nn.Linear(dim_z, 2*dim_Z)
        self.fc22 = nn.Linear(256, 256)
        self.fc23 = nn.Linear(256, 2*dim_Z)


        # Decoder
        self.fc31 = nn.Linear(dim_z + dim_Z, 256)
        #self.fc32 = nn.Linear(256, 256)
        self.fc33 = nn.Linear(256, 28*7*7)
        self.conv31 = torch.nn.ConvTranspose2d(in_channels=28, out_channels=28, kernel_size=4, stride=2, padding=1)
        self.conv32 = torch.nn.ConvTranspose2d(in_channels=28, out_channels=28, kernel_size=4, stride=2, padding=1)
        self.conv33 = torch.nn.ConvTranspose2d(in_channels=28, out_channels=28, kernel_size=4, stride=2, padding=1)
        self.conv34 = torch.nn.ConvTranspose2d(in_channels=28, out_channels=3, kernel_size=4, stride=2, padding=1)


    def encode(self, x):

        h11 = F.relu(self.conv11(x))
        h12 = F.relu(self.conv12(h11))
        h13 = F.relu(self.conv13(h12))
        h14 = F.relu(self.conv14(h13))

        h15 = F.relu(self.fc11(h14))
        h16 = F.relu(self.fc12(h15))
        theta = self.fc13(h16)

        mu = theta[:, :self.dim_z]
        std = torch.exp(0.5 * theta[:, self.dim_z:] )

        return mu, std

    def global_encode(self, z_batch):
        mu = []
        prec = []
        for z in z_batch:
            #h21 = F.relu(self.fc21(z))
            #h22 = F.relu(self.fc22(h21))
            theta = self.fc21(z)
            mu.append(theta[:self.dim_Z])
            #varZ.append(self.fc32(z))
            prec.append(torch.exp(theta[self.dim_Z:])**-1)

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
        h31 = F.relu(self.fc31(aux))
        #h32 = F.relu(self.fc32(h31))
        h33 = F.relu(self.fc33(h31))

        h34 = F.relu(self.conv31(h33.view(-1, 28, 7, 7)))
        #h35 = F.relu(self.conv32(h34))
        #h36 = F.relu(self.conv33(h35))

        return torch.sigmoid(self.conv34(h34))

    def forward(self, x):
        mu_z, var_z = self.encode(x.view(-1, 3, 28, 28))
        z = self.reparameterize(mu_z, var_z)
        mu_Z, var_Z = self.global_encode(z)
        Z = self.reparameterize(mu_Z, var_Z)

        return self.decode(z, Z), mu_z, var_z, mu_Z, var_Z


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, var, mu_g, var_g):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 3*784), x.view(-1, 3*784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + torch.log(var) - mu.pow(2) - var)

    KLG = -0.5 * torch.sum(1 + torch.log(var_g) - mu_g.pow(2) - var_g)

    return BCE + KLD + KLG, BCE, KLD, KLG

