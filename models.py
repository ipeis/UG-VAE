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
from torch.distributions import MultivariateNormal as Normal
from torch.distributions import Categorical as Cat
from torch.distributions import Dirichlet as Dirichlet
from torch.distributions import kl_divergence
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

class GLVAE(nn.Module):
    def __init__(self, channels, dim_z, dim_Z, arch='beta_vae'):
        super(GLVAE, self).__init__()

        self.dim_z = dim_z
        self.dim_Z = dim_Z


        # Architecture from beta_vae
        if arch=='beta_vae':
            # Encoder
            self.pre_encoder = nn.Sequential(
                nn.Conv2d(channels, 32, 4, 2, 1),  # B,  32, 32, 32
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  8,  8
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  4,  4
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 256, 4, 1),  # B, 256,  1,  1
                nn.BatchNorm2d(256),
                nn.ReLU(True),
            )

            self.local_encoder = nn.Sequential(
                View((-1, 256 * 1 * 1)),  # B, 256
                nn.Linear(256, dim_z * 2),  # B, z_dim*2
            )
            self.global_encoder = nn.Sequential(
                View((-1, 256 * 1 * 1)),  # B, 256
                nn.Linear(256, dim_Z * 2),  # B, z_dim*2
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
            self.pre_encoder = nn.Sequential(
                View((-1, channels*784)),
                nn.Linear(channels*784, 400),
                nn.ReLU()
            )
            self.local_encoder = nn.Sequential(
                nn.Linear(400, dim_z*2)
            )
            self.global_encoder = nn.Sequential(
                nn.Linear(400, dim_Z * 2)
            )
            self.decoder = nn.Sequential(
                nn.Linear(dim_z+dim_Z, 400),
                nn.ReLU(),
                nn.Linear(400, channels*784),
                nn.Sigmoid(),
                View((-1, channels, 28, 28))
            )


    def _encode(self, x):
        h = self.pre_encoder(x)
        mu_l, std_l = self._local_encode(h)
        mu_g, std_g = self._global_encode(h)

        return mu_l, std_l, mu_g, std_g


    def _local_encode(self, h):
        theta = self.local_encoder(h)
        mu = theta[:, :self.dim_z]
        var = torch.exp(torch.tanh(theta[:, self.dim_z:]))

        return mu, var


    def _global_encode(self, h):
        theta = self.global_encoder(h)
        mu = theta[:, :self.dim_Z]
        logvar = torch.tanh(theta[:, self.dim_Z:])
        prec = torch.exp(logvar) ** -1

        var = (torch.sum(prec, dim=0))**-1
        aux = torch.sum(torch.mul(prec, mu), dim=0)
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
        mu_l, var_l, mu_g, var_g = self._encode(x)
        z = self.reparameterize(mu_l, var_l)
        Z = self.reparameterize(mu_g, var_g)

        return self._decode(z, Z), mu_l, var_l, mu_g, var_g

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, var, mu_g, var_g, beta_l=1.0, beta_g=1.0, distribution='bernoulli'):

        # BCE = F.binary_cross_entropy(recon_x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]),
        #                             x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]),
        #                             reduction='sum')
        if distribution == 'bernoulli':
            recogn = F.binary_cross_entropy(recon_x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]),
                                            x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]),
                                            reduction='sum')
        elif distribution == 'gaussian':
            recogn = F.mse_loss(recon_x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]),
                                x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + torch.log(var) - mu.pow(2) - var)

        KLG = -0.5 * torch.sum(1 + torch.log(var_g) - mu_g.pow(2) - var_g)

        return recogn + beta_l * KLD + beta_g * KLG, recogn, beta_l * KLD, beta_g * KLG




########################################################################################################

class MGLVAE(nn.Module):
    def __init__(self, channels, dim_z, dim_beta, K, arch='beta_vae', device='cpu'):
        super(MGLVAE, self).__init__()

        self.dim_z = dim_z
        self.dim_beta = dim_beta
        self.K = K
        self.prior_alpha = Dirichlet(torch.ones(1, K) * 10).sample().to(device)
        self.device = device

        # Architecture from beta_vae
        if arch=='beta_vae':
            # Encoder
            self.pre_encoder = nn.Sequential(
                nn.Conv2d(channels, 32, 4, 2, 1),  # B,  32, 32, 32
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  8,  8
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  4,  4
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 256, 4, 1),  # B, 256,  1,  1
                nn.BatchNorm2d(256),
                nn.ReLU(True),
            )

            self.local_encoder = nn.Sequential(
                View((-1, 256)),  # B, 256
                nn.Linear(256, dim_z * 2),  # B, z_dim*2
            )
            self.alpha_encoder = nn.Sequential(
                View((-1, 256 * 1 * 1)),  # B, 256
                nn.Linear(256, K),  # B, K
                nn.Softmax(dim=1)
            )
            self.beta_encoder = nn.Sequential(
                View((-1, 256 + K)),  # B, 256+K
                nn.Linear(256+K, dim_beta * 2),  # B, Z_dim*2
            )
            self.beta_mix = nn.Sequential(
                nn.Linear(K, 256),  # B, 256
                nn.ReLU(True),
                nn.Linear(256, dim_beta * 2),  # B, beta_dim*2
            )
            self.decoder = nn.Sequential(
                nn.Linear(dim_z+dim_beta, 256),  # B, 256
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
            self.pre_encoder = nn.Sequential(
                View((-1, channels*784)),
                nn.Linear(channels*784, 400),
                nn.ReLU()
            )
            self.local_encoder = nn.Sequential(
                nn.Linear(400, dim_z * 2)
            )
            self.alpha_encoder = nn.Sequential(
                nn.Linear(400, K)
            )
            self.beta_encoder = nn.Sequential(
                nn.Linear(400+K, dim_beta * 2)
            )
            self.beta_mix = nn.Sequential(
                nn.Linear(K, 256),  # B, 256
                nn.ReLU(True),
                nn.Linear(256, dim_beta * 2),  # B, beta_dim*2
            )
            self.decoder = nn.Sequential(
                nn.Linear(dim_z+dim_beta, 400),
                nn.ReLU(),
                nn.Linear(400, channels*784),
                nn.Sigmoid(),
                View((-1, channels, 28, 28))
            )


    def _encode(self, x):
        h = self.pre_encoder(x)
        mu_z, var_z = self._local_encode(h)

        p, mu_beta, var_beta = self._global_encode(h)

        return mu_z, var_z, p, mu_beta, var_beta


    def _local_encode(self, h):
        theta = self.local_encoder(h)
        mu = theta[:, :self.dim_z]
        var = torch.exp(torch.tanh(theta[:, self.dim_z:]))

        return mu, var


    def _global_encode(self, h):
        pk_bar = self.alpha_encoder(h)


        #pk = torch.softmax(torch.exp(torch.sum(torch.log(pk_bar), dim=0)), dim=0)

        pk = torch.softmax(torch.mean(pk_bar, dim=0), dim=0)

        input_beta = torch.cat([h.view(h.shape[0], h.shape[1]), pk_bar], dim=1)
        theta = self.beta_encoder(input_beta)
        mu = theta[:, :self.dim_beta]
        logvar = torch.tanh(theta[:, self.dim_beta:])
        prec = torch.exp(logvar) ** -1

        var_beta = (torch.sum(prec, dim=0))**-1
        aux = torch.sum(torch.mul(prec, mu), dim=0)
        mu_beta = torch.mul(var_beta, aux)

        return pk, mu_beta, var_beta

    def _beta_mix(self, k):
        # k one hot
        thetas_beta = self.beta_mix(k)
        mus_beta = thetas_beta[:, :self.dim_beta]
        vars_beta = torch.exp(thetas_beta[:, self.dim_beta:])
        return mus_beta, vars_beta

    def reparameterize(self, mu, var):
        std = var**0.5
        eps = torch.randn_like(std)
        return mu + eps*std

    def _decode(self, z, beta):
        input_decoder = [torch.cat([z[i], beta]) for i in range(len(z))]
        input_decoder = torch.stack(input_decoder)

        return self.decoder(input_decoder)

    def forward(self, x):
        mu_z, var_z, pk, mu_beta, var_beta = self._encode(x)
        z = self.reparameterize(mu_z, var_z)
        beta = self.reparameterize(mu_beta, var_beta)

        mus_beta, vars_beta = self._beta_mix(torch.eye(self.K).to(self.device))

        return self._decode(z, beta), mu_z, var_z, pk, mu_beta, var_beta, mus_beta, vars_beta

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu_z, var_z, pk, mu_beta, var_beta, mus_beta, vars_beta, distribution='bernoulli'):

        # BCE = F.binary_cross_entropy(recon_x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]),
        #                             x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]),
        #                             reduction='sum')
        if distribution == 'bernoulli':
            recogn = F.binary_cross_entropy(recon_x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]),
                                            x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]),
                                            reduction='sum')
        elif distribution == 'gaussian':
            recogn = F.mse_loss(recon_x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]),
                                x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # for non isotropic priors:
        # 0.5 * sum(logVar2 - logVar + torch.exp(logVar) + (mu ** 2 / 2 * sigma2 ** 2) - 0.5)
        #
        KLz = -0.5 * torch.sum(1 + torch.log(var_z) - mu_z.pow(2) - var_z)

        KLbeta=0
        q = Normal(mu_beta, torch.diag(var_beta))
        for mu, var, pk_ in zip(mus_beta, vars_beta, pk):
            p = Normal(mu, torch.diag(var))
            KLbeta+=(pk_ * kl_divergence(q, p).to(self.device))

        q = Cat(pk)
        p = Cat(self.prior_alpha)
        KLalpha = kl_divergence(q, p)

        return recogn + KLz + KLbeta + KLalpha, recogn, KLz, KLbeta, KLalpha



########################################################################################################

class MGLVAEp(nn.Module):
    def __init__(self, channels, dim_z, dim_beta, K, arch='beta_vae', device='cpu'):
        super(MGLVAEp, self).__init__()

        self.dim_z = dim_z
        self.dim_beta = dim_beta
        self.K = K
        self.prior_alpha = Dirichlet(torch.ones(1, K) * 10).sample().to(device)
        self.device = device

        # Architecture from beta_vae
        if arch=='beta_vae':
            # Encoder
            self.pre_encoder = nn.Sequential(
                nn.Conv2d(channels, 32, 4, 2, 1),  # B,  32, 32, 32
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  8,  8
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  4,  4
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 256, 4, 1),  # B, 256,  1,  1
                nn.BatchNorm2d(256),
                nn.ReLU(True),
            )

            self.local_encoder = nn.Sequential(
                View((-1, 256)),  # B, 256
                nn.Linear(256, dim_z * 2),  # B, z_dim*2
            )
            self.alpha_encoder = nn.Sequential(
                View((-1, 256 * 1 * 1)),  # B, 256
                nn.Linear(256, K),  # B, K
                nn.Softmax(dim=1)
            )
            self.beta_encoder = nn.Sequential(
                View((-1, 256 + K)),  # B, 256+K
                nn.Linear(256+K, dim_beta * 2),  # B, Z_dim*2
            )
            self.beta_mix = nn.Sequential(
                nn.Linear(K, 256),  # B, 256
                nn.ReLU(True),
                nn.Linear(256, dim_beta * 2),  # B, beta_dim*2
            )
            self.decoder = nn.Sequential(
                nn.Linear(dim_z+dim_beta, 256),  # B, 256
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
            self.pre_encoder = nn.Sequential(
                View((-1, channels*784)),
                nn.Linear(channels*784, 400),
                nn.ReLU()
            )
            self.local_encoder = nn.Sequential(
                nn.Linear(400, dim_z * 2)
            )
            self.alpha_encoder = nn.Sequential(
                nn.Linear(400, K)
            )
            #self.beta_encoder = nn.Sequential(
            #    nn.Linear(400+K, dim_beta * 2)
            #)
            self.beta_mix = nn.Sequential(
                nn.Linear(K, 256),  # B, 256
                nn.ReLU(True),
                nn.Linear(256, dim_beta * 2),  # B, beta_dim*2
            )
            self.decoder = nn.Sequential(
                nn.Linear(dim_z+dim_beta, 400),
                nn.ReLU(),
                nn.Linear(400, channels*784),
                nn.Sigmoid(),
                View((-1, channels, 28, 28))
            )
        # Special architecture for 32x32 images
        elif arch == 'beta_vae_32':
            # Encoder
            self.pre_encoder = nn.Sequential(
                #nn.Conv2d(channels, 32, 4, 2, 1),  # B,  32, 32, 32
                #nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(channels, 32, 4, 2, 1),  # B,  32, 16, 16
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  8,  8
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  4,  4
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 256, 4, 1),  # B, 256,  1,  1
                nn.BatchNorm2d(256),
                nn.ReLU(True),
            )

            self.local_encoder = nn.Sequential(
                View((-1, 256)),  # B, 256
                nn.Linear(256, dim_z * 2),  # B, z_dim*2
            )
            self.alpha_encoder = nn.Sequential(
                View((-1, 256 * 1 * 1)),  # B, 256
                nn.Linear(256, K),  # B, K
                nn.Softmax(dim=1)
            )
            self.beta_encoder = nn.Sequential(
                View((-1, 256 + K)),  # B, 256+K
                nn.Linear(256 + K, dim_beta * 2),  # B, Z_dim*2
            )
            self.beta_mix = nn.Sequential(
                nn.Linear(K, 256),  # B, 256
                nn.ReLU(True),
                nn.Linear(256, dim_beta * 2),  # B, beta_dim*2
            )
            self.decoder = nn.Sequential(
                nn.Linear(dim_z + dim_beta, 256),  # B, 256
                View((-1, 256, 1, 1)),  # B, 256,  1,  1
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 64, 4),  # B,  64,  4,  4
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 64, 4, 2, 1),  # B,  64,  8,  8
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  32, 16, 16
                nn.ReLU(True),
                nn.ConvTranspose2d(32, channels, 4, 2, 1),  # B,  32, 32, 32
                #nn.ReLU(True),
                #nn.ConvTranspose2d(32, channels, 4, 2, 1),  # B, nc, 64, 64
                nn.Sigmoid()
            )



    def _encode(self, x):
        h = self.pre_encoder(x)
        mu_z, var_z = self._local_encode(h)

        p, mus_beta, vars_beta = self._global_encode(h)

        return mu_z, var_z, p, mus_beta, vars_beta


    def _local_encode(self, h):
        theta = self.local_encoder(h)
        mu = theta[:, :self.dim_z]
        var = torch.exp(torch.tanh(theta[:, self.dim_z:]))

        return mu, var


    def _global_encode(self, h):
        pk_bar = self.alpha_encoder(h)


        #pk = torch.softmax(torch.exp(torch.sum(torch.log(pk_bar), dim=0)), dim=0)

        pk = torch.softmax(torch.mean(pk_bar, dim=0), dim=0)

        mus_beta, vars_beta = self._beta_mix(torch.eye(self.K).to(self.device))

        """
        input_beta = torch.cat([h.view(h.shape[0], h.shape[1]), pk_bar], dim=1)
        theta = self.beta_encoder(input_beta)
        mu = theta[:, :self.dim_beta]
        logvar = torch.tanh(theta[:, self.dim_beta:])
        prec = torch.exp(logvar) ** -1

        var_beta = (torch.sum(prec, dim=0))**-1
        aux = torch.sum(torch.mul(prec, mu), dim=0)
        mu_beta = torch.mul(var_beta, aux)
        """

        return pk, mus_beta, vars_beta

    def _beta_mix(self, k):
        # k one hot
        thetas_beta = self.beta_mix(k)
        mus_beta = thetas_beta[:, :self.dim_beta]
        vars_beta = torch.exp(thetas_beta[:, self.dim_beta:])
        return mus_beta, vars_beta

    def reparameterize(self, mu, var):
        std = var**0.5
        eps = torch.randn_like(std)
        return mu + eps*std

    def _decode(self, z, beta):
        input_decoder = [torch.cat([z[i], beta]) for i in range(len(z))]
        input_decoder = torch.stack(input_decoder)

        return self.decoder(input_decoder)

    def forward(self, x):
        mu_z, var_z, pk, mus_beta, vars_beta = self._encode(x)
        z = self.reparameterize(mu_z, var_z)
        betas = [self.reparameterize(mu_beta, var_beta) for mu_beta, var_beta in zip(mus_beta, vars_beta)]
        recon = [self._decode(z, beta) for beta in betas]
        mu_beta = mus_beta
        var_beta = vars_beta

        #mus_beta, vars_beta = self._beta_mix(torch.eye(self.K).to(self.device))

        return recon, mu_z, var_z, pk, mu_beta, var_beta

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu_z, var_z, pk,
                      distribution='bernoulli'):

        # BCE = F.binary_cross_entropy(recon_x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]),
        #                             x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]),
        #                             reduction='sum')
        if distribution == 'bernoulli':
            recogn = torch.stack([p*F.binary_cross_entropy(rec_x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]),
                                                x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]),
                                                reduction='sum') for rec_x, p in zip(recon_x, pk)])
            recogn = torch.sum(recogn)

        elif distribution == 'gaussian':
            recogn = torch.stack([p*F.mse_loss(rec_x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]),
                                 x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]), reduction='sum')
                      for rec_x,p in zip(recon_x, pk)])
            recogn = torch.sum(recogn)

                # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # for non isotropic priors:
        # 0.5 * sum(logVar2 - logVar + torch.exp(logVar) + (mu ** 2 / 2 * sigma2 ** 2) - 0.5)
        #
        KLz = -0.5 * torch.sum(1 + torch.log(var_z) - mu_z.pow(2) - var_z)
        """
        KLbeta=0
        q = Normal(mu_beta, torch.diag(var_beta))
        for mu, var, pk_ in zip(mus_beta, vars_beta, pk):
            p = Normal(mu, torch.diag(var))
            KLbeta+=(pk_ * kl_divergence(q, p).to(self.device))
        """
        q = Cat(pk)
        p = Cat(self.prior_alpha)
        KLalpha = kl_divergence(q, p)

        return recogn + KLz + KLalpha, recogn, KLz, KLalpha


class GGMVAE(nn.Module):
    def __init__(self, channels, dim_z, dim_beta, dim_w, K, arch='beta_vae', device='cpu'):
        super(GGMVAE, self).__init__()

        self.dim_z = dim_z
        self.dim_beta = dim_beta
        self.dim_w = dim_w
        self.K = K
        #self.pi_apha = Dirichlet(torch.ones(1, K) * 10).sample().to(device)
        self.pi_alpha = torch.ones(1, K) / K
        self.device = device

        # Architecture from beta_vae
        if arch=='beta_vae':
            # Encoder
            self.pre_encoder = nn.Sequential(
                nn.Conv2d(channels, 32, 4, 2, 1),  # B,  32, 32, 32
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  8,  8
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  4,  4
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 256, 4, 1),  # B, 256,  1,  1
                nn.BatchNorm2d(256),
                nn.ReLU(True),
            )
            self.decoder = nn.Sequential(
                nn.Linear(dim_z+dim_beta, 256),  # B, 256
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
            self.pre_encoder = nn.Sequential(
                View((-1, channels*784)),
                nn.Linear(channels*784, 256),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(dim_z+dim_beta, 256),
                nn.ReLU(),
                nn.Linear(256, channels*784),
                nn.Sigmoid(),
                View((-1, channels, 28, 28))
            )
        self.encoder_z = nn.Sequential(
            View((-1, 256)),  # B, 256
            nn.Linear(256, dim_z * 2),  # B, dim_z*2
        )
        self.encoder_beta = nn.Sequential(
            View((-1, 256)),  # B, 256
            nn.Linear(256, dim_beta * 2),  # B, dim_beta*2
        )
        self.encoder_w = nn.Sequential(
            View((-1, 256)),  # B, 256
            nn.Linear(256, dim_w * 2),  # B, dim_w*2
        )
        self.encoder_alpha = nn.Sequential(
            # View((-1, dim_beta+dim_w)),  # B, dim_beta+dim_w
            nn.Linear(dim_beta + dim_w, 256),  # B, 256
            nn.ReLU(True),
            nn.Linear(256, K),  # B, K
            nn.Softmax(dim=0)
        )
        self.beta_gen = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_w, 256),  # B, 256
                nn.ReLU(True),
                # nn.Linear(256, 256),  # B, 256
                # nn.ReLU(True),
                nn.Linear(256, dim_beta*2)
            )
            for k in range(self.K)])


    def gaussian_prod(self, mus, logvars):

        prec = torch.exp(logvars) ** -1

        var = (torch.sum(prec, dim=0)) ** -1
        aux = torch.sum(torch.mul(prec, mus), dim=0)
        mu = torch.mul(var, aux)

        return mu, var

    def _encode_z(self, h):
        out = self.encoder_z(h)
        mu = out[:, :self.dim_z]
        var = torch.exp(torch.tanh(out[:, self.dim_z:]))
        return mu, var

    def _encode_beta(self, h):
        out = self.encoder_beta(h)
        mus = out[:, :self.dim_beta]
        logvars = torch.tanh(out[:, self.dim_beta:])
        mu, var = self.gaussian_prod(mus, logvars)
        return mu, var

    def _encode_w(self, h):
        out = self.encoder_w(h)
        mus = out[:, :self.dim_w]
        logvars = torch.tanh(out[:, self.dim_w:])
        mu, var = self.gaussian_prod(mus, logvars)
        return mu, var

    def _encode_alpha(self, w, beta):
        input = torch.cat([w, beta], dim=0)
        pi = self.encoder_alpha(input)
        return pi

    def reparameterize(self, mu, var):
        std = var**0.5
        eps = torch.randn_like(std)
        return mu + eps*std

    def _beta_gen(self, w):
        mus = []
        vars = []
        for k in range(self.K):
            out = self.beta_gen[k](w)
            mu = out[:self.dim_beta]
            var = torch.exp(torch.tanh(out[self.dim_beta:]))
            mus.append(mu)
            vars.append(var)
        mus = torch.stack(mus)
        vars = torch.stack(vars)
        return mus, vars

    def _decode(self, z, beta):
        input_decoder = [torch.cat([z[i], beta]) for i in range(len(z))]
        input_decoder = torch.stack(input_decoder)

        return self.decoder(input_decoder)

    def forward(self, x):

        # Encode
        h = self.pre_encoder(x)
        mu_z, var_z = self._encode_z(h)
        z = self.reparameterize(mu_z, var_z)
        mu_beta, var_beta = self._encode_beta(h)
        mu_w, var_w = self._encode_w(h)
        beta = self.reparameterize(mu_beta, var_beta)
        w = self.reparameterize(mu_w, var_w)
        pi = self._encode_alpha(beta, w)

        # Decode
        mu_x = self._decode(z, beta)

        # Generative params
        mus_beta_p, vars_beta_p = self._beta_gen(w)

        return mu_x, mu_z, var_z, mu_beta, var_beta, mu_w, var_w, pi, mus_beta_p, vars_beta_p

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu_z, var_z, mu_beta, var_beta, mu_w, var_w, pi, mus_beta, vars_beta, distribution='bernoulli'):

        # BCE = F.binary_cross_entropy(recon_x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]),
        #                             x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]),
        #                             reduction='sum')
        if distribution == 'bernoulli':
            recogn = F.binary_cross_entropy(recon_x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]),
                                            x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]),
                                            reduction='sum')
        elif distribution == 'gaussian':
            recogn = F.mse_loss(recon_x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]),
                                x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # for non isotropic priors:
        # 0.5 * sum(logVar2 - logVar + torch.exp(logVar) + (mu ** 2 / 2 * sigma2 ** 2) - 0.5)
        #

        KLz = -0.5 * torch.sum(1 + torch.log(var_z) - mu_z.pow(2) - var_z)

        # Given p(alpha)=Cat(1/K): KLalpha = -log(K) -sum(log(pi_k))/K
        KLalpha = -torch.log(torch.tensor(self.K, dtype=torch.float).to(self.device)) - torch.sum(torch.log(pi)) / self.K
        """
        q = Cat(pi)
        p = Cat(self.prior_alpha)
        KLalpha = kl_divergence(q, p)
        KLz = -0.5 * torch.sum(1 + torch.log(var_z) - mu_z.pow(2) - var_z)
        """

        KLbeta=0
        q = Normal(mu_beta, torch.diag(var_beta))
        for mu, var, pi_k in zip(mus_beta, vars_beta, pi):
            p = Normal(mu, torch.diag(var))
            KLbeta+=(pi_k * kl_divergence(q, p).to(self.device))

        KLw = -0.5 * torch.sum(1 + torch.log(var_w) - mu_w.pow(2) - var_w)

        #return recogn + KLz + KLbeta + KLalpha + KLw, recogn, KLz, KLbeta, KLalpha, KLw
        return recogn, recogn, KLz, KLbeta, KLalpha, KLw

class GGMVAE2(nn.Module):
    def __init__(self, channels, dim_z, dim_beta, dim_w, K, arch='beta_vae', device='cpu'):
        super(GGMVAE2, self).__init__()

        self.dim_z = dim_z
        self.dim_beta = dim_beta
        self.dim_w = dim_w
        self.K = K
        #self.pi_apha = Dirichlet(torch.ones(1, K) * 10).sample().to(device)
        self.pi_alpha = torch.ones(1, K) / K
        self.device = device

        # Architecture from beta_vae
        if arch=='beta_vae':
            # Encoder
            self.pre_encoder = nn.Sequential(
                nn.Conv2d(channels, 32, 4, 2, 1),  # B,  32, 32, 32
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  8,  8
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  4,  4
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 256, 4, 1),  # B, 256,  1,  1
                nn.BatchNorm2d(256),
                nn.ReLU(True),
            )
            self.decoder = nn.Sequential(
                nn.Linear(dim_z+dim_beta, 256),  # B, 256
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
            self.pre_encoder = nn.Sequential(
                View((-1, channels*784)),
                nn.Linear(channels*784, 256),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(dim_z+dim_beta, 256),
                nn.ReLU(),
                nn.Linear(256, channels*784),
                nn.Sigmoid(),
                View((-1, channels, 28, 28))
            )
        self.encoder_z = nn.Sequential(
            View((-1, 256)),  # B, 256
            nn.Linear(256, dim_z * 2),  # B, dim_z*2
        )
        """
        self.encoder_beta = nn.Sequential(
            View((-1, 256)),  # B, 256
            nn.Linear(256, dim_beta * 2),  # B, dim_beta*2
        )
        """
        self.encoder_w = nn.Sequential(
            View((-1, 256)),  # B, 256
            nn.Linear(256, dim_w * 2),  # B, dim_w*2
        )
        self.encoder_alpha = nn.Sequential(
            # View((-1, dim_beta+dim_w)),  # B, dim_beta+dim_w
            nn.Linear(256 + dim_w, 256),  # B, 256
            nn.Tanh(),
            nn.Linear(256, K),  # B, K
            nn.Softmax(dim=0)
        )
        self.encoder_alpha_X = nn.Sequential(
            View((-1, 256)),
            nn.Linear(256, 256),
        )
        self.beta_gen = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_w, 256),  # B, 256
                nn.ReLU(True),
                # nn.Linear(256, 256),  # B, 256
                # nn.ReLU(True),
                nn.Linear(256, dim_beta*2)
            )
            for k in range(self.K)])


    def gaussian_prod(self, mus, logvars):

        prec = torch.exp(logvars) ** -1

        var = (torch.sum(prec, dim=0)) ** -1
        aux = torch.sum(torch.mul(prec, mus), dim=0)
        mu = torch.mul(var, aux)

        return mu, var

    def _encode_z(self, h):
        out = self.encoder_z(h)
        mu = out[:, :self.dim_z]
        var = torch.exp(torch.tanh(out[:, self.dim_z:]))
        return mu, var

    def _encode_beta(self, h):
        out = self.encoder_beta(h)
        mus = out[:, :self.dim_beta]
        logvars = torch.tanh(out[:, self.dim_beta:])
        mu, var = self.gaussian_prod(mus, logvars)
        return mu, var

    def _encode_w(self, h):
        out = self.encoder_w(h)
        mus = out[:, :self.dim_w]
        logvars = torch.tanh(out[:, self.dim_w:])
        mu, var = self.gaussian_prod(mus, logvars)
        return mu, var

    def _encode_alpha(self, H, w):

        h = torch.mean(self.encoder_alpha_X(H), dim=0)
        input = torch.cat([h, w], dim=0)
        pi = self.encoder_alpha(input)
        return pi

    def reparameterize(self, mu, var):
        std = var**0.5
        eps = torch.randn_like(std)
        return mu + eps*std

    def _beta_gen(self, w):
        mus = []
        vars = []
        for k in range(self.K):
            out = self.beta_gen[k](w)
            mu = out[:self.dim_beta]
            var = torch.exp(torch.tanh(out[self.dim_beta:]))
            mus.append(mu)
            vars.append(var)
        mus = torch.stack(mus)
        vars = torch.stack(vars)
        return mus, vars

    def _decode(self, z, beta):
        input_decoder = [torch.cat([z[i], beta]) for i in range(len(z))]
        input_decoder = torch.stack(input_decoder)

        return self.decoder(input_decoder)

    def forward(self, x):

        # Encode
        h = self.pre_encoder(x)
        mu_z, var_z = self._encode_z(h)
        z = self.reparameterize(mu_z, var_z)
        #mu_beta, var_beta = self._encode_beta(h)
        mu_w, var_w = self._encode_w(h)
        w = self.reparameterize(mu_w, var_w)
        mus_beta, vars_beta = self._beta_gen(w)
        betas = [self.reparameterize(mu_beta, var_beta) for mu_beta, var_beta in zip(mus_beta, vars_beta)]
        pi = self._encode_alpha(h, w)

        # Decode
        mus_x = [self._decode(z, beta) for beta in betas]

        return mus_x, mu_z, var_z, mus_beta, vars_beta, mu_w, var_w, pi


    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, mus_x, x, mu_z, var_z, mus_beta, vars_beta, mu_w, var_w, pi, distribution='bernoulli'):

        # BCE = F.binary_cross_entropy(recon_x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]),
        #                             x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]),
        #                             reduction='sum')

        recogn=0
        if distribution == 'bernoulli':
            for i, mu_x in enumerate(mus_x):
                recogn += pi[i] * F.binary_cross_entropy(mu_x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]),
                                            x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]),
                                            reduction='sum')
        elif distribution == 'gaussian':
            for i, mu_x in enumerate(mus_x):
                recogn += pi[i] * F.mse_loss(mu_x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]),
                                x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # for non isotropic priors:
        # 0.5 * sum(logVar2 - logVar + torch.exp(logVar) + (mu ** 2 / 2 * sigma2 ** 2) - 0.5)
        #

        KLz = -0.5 * torch.sum(1 + torch.log(var_z) - mu_z.pow(2) - var_z)

        # Given p(alpha)=Cat(1/K): KLalpha = -log(K) -sum(log(pi_k))/K
        KLalpha = -torch.log(torch.tensor(self.K, dtype=torch.float).to(self.device)) - torch.sum(torch.log(pi)) / self.K

        """
        q = Cat(pi)
        p = Cat(self.prior_alpha)
        KLalpha = kl_divergence(q, p)
        KLz = -0.5 * torch.sum(1 + torch.log(var_z) - mu_z.pow(2) - var_z)
        """
        """
        KLbeta=0
        q = Normal(mu_beta, torch.diag(var_beta))
        for mu, var, pi_k in zip(mus_beta, vars_beta, pi):
            p = Normal(mu, torch.diag(var))
            KLbeta+=(pi_k * kl_divergence(q, p).to(self.device))
        """

        KLw = -0.5 * torch.sum(1 + torch.log(var_w) - mu_w.pow(2) - var_w)

        return recogn + KLz + KLalpha + KLw, recogn, KLz, KLalpha, KLw
        #return recogn, recogn, KLz, KLalpha, KLw



class GGMVAE3(nn.Module):
    def __init__(self, channels, dim_z, dim_beta, dim_w, K, arch='beta_vae', device='cpu'):
        super(GGMVAE3, self).__init__()

        self.dim_z = dim_z
        self.dim_beta = dim_beta
        self.dim_w = dim_w
        self.K = K
        #self.pi_apha = Dirichlet(torch.ones(1, K) * 10).sample().to(device)
        self.pi_alpha = torch.ones(1, K) / K
        self.device = device

        # Architecture from beta_vae
        if arch=='beta_vae':
            # Encoder
            self.pre_encoder = nn.Sequential(
                nn.Conv2d(channels, 32, 4, 2, 1),  # B,  32, 32, 32
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  8,  8
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  4,  4
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 256, 4, 1),  # B, 256,  1,  1
                nn.BatchNorm2d(256),
                nn.ReLU(True),
            )
            self.decoder = nn.Sequential(
                nn.Linear(dim_z+dim_beta, 256),  # B, 256
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
            self.pre_encoder = nn.Sequential(
                View((-1, channels*784)),
                nn.Linear(channels*784, 256),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(dim_z+dim_beta, 256),
                nn.ReLU(),
                nn.Linear(256, channels*784),
                nn.Sigmoid(),
                View((-1, channels, 28, 28))
            )
        self.encoder_z = nn.Sequential(
            View((-1, 256)),  # B, 256
            nn.Linear(256, dim_z * 2),  # B, dim_z*2
        )
        """
        self.encoder_beta = nn.Sequential(
            View((-1, 256)),  # B, 256
            nn.Linear(256, dim_beta * 2),  # B, dim_beta*2
        )
        """
        self.encoder_w = nn.Sequential(
            View((-1, 256)),  # B, 256
            nn.Linear(256, dim_w * 2),  # B, dim_w*2
        )
        self.encoder_alpha = nn.Sequential(
            # View((-1, dim_beta+dim_w)),  # B, dim_beta+dim_w
            nn.Linear(256 + dim_w, 256),  # B, 256
            nn.Tanh(),
            nn.Linear(256, K),  # B, K
            nn.Softmax(dim=0)
        )
        self.encoder_alpha_X = nn.Sequential(
            View((-1, 256)),
            nn.Linear(256, 256),
        )
        self.beta_gen = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_w, 256),  # B, 256
                nn.ReLU(True),
                # nn.Linear(256, 256),  # B, 256
                # nn.ReLU(True),
                nn.Linear(256, dim_beta*2)
            )
            for k in range(self.K)])

        self.z_prior = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_beta, 256),  # B, 256
                nn.ReLU(True),
                # nn.Linear(256, 256),  # B, 256
                # nn.ReLU(True),
                nn.Linear(256, dim_z * 2)
            )
            for k in range(self.K)])

    def gaussian_prod(self, mus, logvars):

        prec = torch.exp(logvars) ** -1

        var = (torch.sum(prec, dim=0)) ** -1
        aux = torch.sum(torch.mul(prec, mus), dim=0)
        mu = torch.mul(var, aux)

        return mu, var

    def _encode_z(self, h):
        out = self.encoder_z(h)
        mu = out[:, :self.dim_z]
        var = torch.exp(torch.tanh(out[:, self.dim_z:]))
        return mu, var

    def _encode_w(self, h):
        out = self.encoder_w(h)
        mus = out[:, :self.dim_w]
        logvars = torch.tanh(out[:, self.dim_w:])
        mu, var = self.gaussian_prod(mus, logvars)
        return mu, var

    def _encode_alpha(self, H, w):

        h = torch.mean(self.encoder_alpha_X(H), dim=0)
        input = torch.cat([h, w], dim=0)
        pi = self.encoder_alpha(input)
        return pi

    def reparameterize(self, mu, var):
        std = var**0.5
        eps = torch.randn_like(std)
        return mu + eps*std

    def _beta_gen(self, w):
        mus = []
        vars = []
        for k in range(self.K):
            out = self.beta_gen[k](w)
            mu = out[:self.dim_beta]
            var = torch.exp(torch.tanh(out[self.dim_beta:]))
            mus.append(mu)
            vars.append(var)
        mus = torch.stack(mus)
        vars = torch.stack(vars)
        return mus, vars

    def _z_prior(self, betas):

        out = [self.z_prior[k](betas[k]) for k in range(self.K)]

        mus_z = torch.stack([out[k][:self.dim_z] for k in range(self.K)])
        vars_z = torch.stack([torch.exp(torch.tanh(out[k][self.dim_z:])) for k in range(self.K)])

        return mus_z, vars_z

    def _decode(self, z, beta):
        input_decoder = [torch.cat([z[i], beta]) for i in range(len(z))]
        input_decoder = torch.stack(input_decoder)

        return self.decoder(input_decoder)

    def forward(self, x):

        # Encode
        h = self.pre_encoder(x)
        mu_z, var_z = self._encode_z(h)
        z = self.reparameterize(mu_z, var_z)
        mu_w, var_w = self._encode_w(h)
        w = self.reparameterize(mu_w, var_w)
        mus_beta, vars_beta = self._beta_gen(w)
        betas = [self.reparameterize(mu_beta, var_beta) for mu_beta, var_beta in zip(mus_beta, vars_beta)]
        pi = self._encode_alpha(h, w)

        mus_z_p, vars_z_p = self._z_prior(betas)

        # Decode
        mus_x = [self._decode(z, beta) for beta in betas]

        return mus_x, mu_z, var_z, mus_beta, vars_beta, mus_z_p, vars_z_p, mu_w, var_w, pi


    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, mus_x, x, mu_z, var_z, mus_z_p, vars_z_p, mu_w, var_w, pi, distribution='bernoulli'):

        recogn=0
        if distribution == 'bernoulli':
            for i, mu_x in enumerate(mus_x):
                recogn += pi[i] * F.binary_cross_entropy(mu_x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]),
                                            x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]),
                                            reduction='sum')
        elif distribution == 'gaussian':
            for i, mu_x in enumerate(mus_x):
                recogn += pi[i] * F.mse_loss(mu_x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]),
                                x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # for non isotropic priors:
        # 0.5 * sum(logVar2 - logVar + torch.exp(logVar) + (mu ** 2 / 2 * sigma2 ** 2) - 0.5)
        #

        KLz = 0
        for mu, var, pi_k in zip(mus_z_p, vars_z_p, pi):
            p = Normal(mu, torch.diag(var))
            for mu_z_q, var_z_q in zip(mu_z, var_z):
                q = Normal(mu_z_q, torch.diag(var_z_q))
                KLz += (pi_k * kl_divergence(q, p).to(self.device))

        #KLz = -0.5 * torch.sum(1 + torch.log(var_z) - mu_z.pow(2) - var_z)

        # Given p(alpha)=Cat(1/K): KLalpha = -log(K) -sum(log(pi_k))/K
        KLalpha = -torch.log(torch.tensor(self.K, dtype=torch.float).to(self.device)) - torch.sum(torch.log(pi)) / self.K


        KLw = -0.5 * torch.sum(1 + torch.log(var_w) - mu_w.pow(2) - var_w)

        return recogn + KLz + KLalpha + KLw, recogn, KLz, KLalpha, KLw



class betaVAE(nn.Module):
    def __init__(self, channels, dim_z, arch='beta_vae'):
        super(betaVAE, self).__init__()

        self.dim_z = dim_z

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
                nn.Linear(dim_z, 256),  # B, 256
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
                nn.Linear(dim_z, 400),
                nn.ReLU(),
                nn.Linear(400, channels*784),
                nn.Sigmoid(),
                View((-1, channels, 28, 28))
            )

    def _encode(self, x):
        theta = self.encoder(x)
        mu = theta[:, :self.dim_z]
        std = torch.exp(0.5 * theta[:, self.dim_z:] )

        return mu, std


    def reparameterize(self, mu, var):
        std = var**0.5
        eps = torch.randn_like(std)
        return mu + eps*std

    def _decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu_z, var_z = self._encode(x)
        z = self.reparameterize(mu_z, var_z)

        return self._decode(z), mu_z, var_z


    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, var, beta=1.0, distribution='bernoulli'):

        #BCE = F.binary_cross_entropy(recon_x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]),
        #                             x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]),
        #                             reduction='sum')
        if distribution=='bernoulli':
            recogn = F.binary_cross_entropy(recon_x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]),
                                     x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]),
                                            reduction='sum')
        if distribution=='gaussian':
            recogn = F.mse_loss(recon_x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]),
                                x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + torch.log(var) - mu.pow(2) - var)

        return recogn + beta*KLD, recogn, beta*KLD





class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
