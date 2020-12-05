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

class MLVAE(nn.Module):
    # C: global variable
    # s: local variable
    def __init__(self, channels, dim_s, dim_C, arch='beta_vae'):
        super(MLVAE, self).__init__()

        self.dim_s = dim_s
        self.dim_C = dim_C


        # Architecture from beta_vae
        if arch=='beta_vae':
            # Encoder
            self.pre_encoder = nn.Sequential(
                nn.Conv2d(channels, 32, 4, 2, 1),  # B,  32, 32, 32
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
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
                nn.Linear(256, dim_s * 2),  # B, z_dim*2
            )
            self.global_encoder = nn.Sequential(
                View((-1, 256 * 1 * 1)),  # B, 256
                nn.Linear(256, dim_C * 2),  # B, z_dim*2
            )

            self.decoder = nn.Sequential(
                nn.Linear(dim_s+dim_C, 256),  # B, 256
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
                nn.Linear(400, dim_s*2)
            )
            self.global_encoder = nn.Sequential(
                nn.Linear(400, dim_C * 2)
            )
            self.decoder = nn.Sequential(
                nn.Linear(dim_s+dim_C, 400),
                nn.ReLU(),
                nn.Linear(400, channels*784),
                nn.Sigmoid(),
                View((-1, channels, 28, 28))
            )


    def _encode(self, x):
        h = self.pre_encoder(x)
        mu_s, std_s = self._local_encode(h)
        mu_C, std_C = self._global_encode(h)

        return mu_s, std_s, mu_C, std_C


    def _local_encode(self, h):
        theta = self.local_encoder(h)
        mu = theta[:, :self.dim_s]
        var = torch.exp(torch.tanh(theta[:, self.dim_s:]))

        return mu, var


    def _global_encode(self, h):
        theta = self.global_encoder(h)
        mu = theta[:, :self.dim_C]
        logvar = torch.tanh(theta[:, self.dim_C:])
        prec = torch.exp(logvar) ** -1

        var = (torch.sum(prec, dim=0))**-1
        aux = torch.sum(torch.mul(prec, mu), dim=0)
        mu = torch.mul(var, aux)

        return mu, var

    def reparameterize(self, mu, var):
        std = var**0.5
        eps = torch.randn_like(std)
        return mu + eps*std

    def _decode(self, s, C):
        sC = [torch.cat([s[i], C]) for i in range(len(s))]
        sC = torch.stack(sC)

        return self.decoder(sC)

    def forward(self, x):
        mu_s, var_s, mu_C, var_C = self._encode(x)
        s = self.reparameterize(mu_s, var_s)
        C = self.reparameterize(mu_C, var_C)

        return self._decode(s, C), mu_s, var_s, mu_C, var_C

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu_s, var_s, mu_C, var_C, distribution='gaussian'):

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
        KLs = -0.5 * torch.sum(1 + torch.log(var_s) - mu_s.pow(2) - var_s)

        KLC = -0.5 * torch.sum(1 + torch.log(var_C) - mu_C.pow(2) - var_C)

        return recogn + KLs + KLC, recogn, KLs, KLC



########################################################################################################


class UGVAE(nn.Module):
    # This model is a GGMVAE in the local space, where the noise come from the global space∫
    def __init__(self, channels, dim_z, dim_beta, K, var_x=0.1, arch='beta_vae', device='cpu'):
        super(UGVAE, self).__init__()

        self.dim_z = dim_z
        self.dim_beta = dim_beta
        self.K = K
        self.var_x=var_x
        self.pi_d = torch.ones(1, K) / K
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
                nn.Linear(256, channels * 784),
                nn.Sigmoid(),
                View((-1, channels, 28, 28))
            )

        self.encoder_z = nn.Sequential(
            View((-1, 256)),  # B, 256
            nn.Linear(256, dim_z * 2),  # B, dim_z*2
        )

        self.encoder_beta = nn.Sequential(
            View((-1, 256+K)),  # B, 256
            nn.Linear(256+K, dim_beta * 2),  # B, dim_beta*2
        )

        self.encoder_d = nn.Sequential(
            # View((-1, dim_beta+dim_w)),  # B, dim_beta+dim_w
            nn.Linear(dim_z, 256),  # B, 256
            nn.Tanh(),
            nn.Linear(256, K),  # B, K
            #StableSofmax(L)
            nn.Softmax(dim=1)
        )

        self.z_prior = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_beta, 256),  # B, 256
                nn.ReLU(True),
                # nn.Linear(256, 256),  # B, 256
                # nn.ReLU(True),
                nn.Linear(256, dim_z * 2)
            )
            for k in range(K)])

    def gaussian_prod(self, mus, logvars):

        prec = torch.exp(logvars) ** -1

        var = (torch.sum(prec, dim=0)) ** -1
        aux = torch.sum(torch.mul(prec, mus), dim=0)
        mu = torch.mul(var, aux)

        return mu, var

    def _encode_z(self, h):
        out = self.encoder_z(h)
        mu = out[:, :self.dim_z]
        var = torch.exp(out[:, self.dim_z:])
        return mu, var

    def _encode_beta(self, h, pi):
        input = torch.cat([h.view(-1, 256), pi], dim=1)
        out = self.encoder_beta(input)
        mus = out[:, :self.dim_beta]
        logvars = out[:, self.dim_beta:]
        mu, var = self.gaussian_prod(mus, logvars)
        return mu, var

    def _encode_d(self, z):
        pi = self.encoder_d(z) + 1e-20 # add a constant to avoid log(0)
        return pi

    def reparameterize(self, mu, var):
        std = var**0.5
        eps = torch.randn_like(std)
        return mu + eps*std

    def _z_prior(self, beta):
        out = [self.z_prior[k](beta) for k in range(self.K)]
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
        pi = self._encode_d(z)
        mu_beta, var_beta = self._encode_beta(h, pi)
        beta = self.reparameterize(mu_beta, var_beta)
        mus_z, vars_z = self._z_prior(beta)

        # Decode
        mu_x = self._decode(z, beta)

        return mu_x, mu_z, var_z, mus_z, vars_z, mu_beta, var_beta, pi


    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, x, mu_x, mu_z, var_z, mus_z, vars_z, mu_beta, var_beta, pi):

        # logp() for a multivariate gaussian with diagonal cov
        D = x.shape[-1]*x.shape[-2]     # Dimension of the image
        x = x.reshape(-1, 1, D)
        mu_x = mu_x.reshape(-1, 1, D)
        var_x = torch.ones_like(mu_x) * self.var_x
        cnt = D*np.log(2*np.pi)+torch.sum(torch.log(var_x), dim=-1)
        logp = torch.sum( -0.5 * (cnt + torch.sum((x-mu_x)*var_x**-1*(x-mu_x), dim=-1)) )

        KLz = 0
        l=0
        for mu, var in zip(mus_z, vars_z):
            KLz += ( 0.5 * torch.sum( torch.unsqueeze(pi[:, l], 1) * (- 1 + var**-1 * var_z + (mu-mu_z)**2*var**-1
                            +torch.log(var) - torch.log(var_z))))
            l+=1

        KLbeta = -0.5 * torch.sum(1 + torch.log(var_beta) - mu_beta.pow(2) - var_beta)

        KLd = torch.sum(pi*(torch.log(pi.double()).float()+ torch.log(torch.tensor(self.K, dtype=torch.float).to(self.device)) ))

        # To maximize ELBO we minimize loss (-ELBO)
        return -logp + KLz + KLd + KLbeta, -logp, KLz, KLd, KLbeta


    def logmeanexp(self, inputs, dim=1):
        if inputs.size(dim) == 1:
            return inputs
        else:
            input_max = inputs.max(dim, keepdim=True)[0]
            return (inputs - input_max).exp().mean(dim).log() + input_max


    def loss_iwae(self, S, x, mu_x, mu_z, var_z, mus_z, vars_z, mu_beta, var_beta, pi):

        elbo = torch.cat(self.loss_function(x, mu_x, mu_z, var_z, mus_z, vars_z, mu_beta, var_beta, pi))

        self.logmeanexp(elbo, 1).squeeze(1)  # mean_n, batch_size

        # p(X | Z, beta)
        D = x.shape[-1] * x.shape[-2]  # Dimension of the image
        x = x.reshape(-1, 1, D)
        mu_x = mu_x.reshape(-1, 1, D)
        var_x = torch.ones_like(mu_x) * self.var_x
        cnt = D * np.log(2 * np.pi) + torch.sum(torch.log(var_x), dim=-1)
        logp = torch.sum(-0.5 * (cnt + torch.sum((x - mu_x) * var_x ** -1 * (x - mu_x), dim=-1)))
        p = torch.exp(logp)

        # p(Z | beta, D)


class betaVAE(nn.Module):
    def __init__(self, channels, dim_z, var_x=0.1, arch='beta_vae'):
        super(betaVAE, self).__init__()

        self.dim_z = dim_z
        self.var_x = var_x

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
    def loss_function(self, x, mu_x, mu_z, var_z, beta=1.0):

        # logp() for a multivariate gaussian with diagonal cov
        D = x.shape[-1] * x.shape[-2]  # Dimension of the image
        x = x.reshape(-1, 1, D)
        mu_x = mu_x.reshape(-1, 1, D)
        var_x = torch.ones_like(mu_x) * self.var_x
        cnt = D * np.log(2 * np.pi) + torch.sum(torch.log(var_x), dim=-1)
        logp = torch.sum(-0.5 * (cnt + torch.sum((x - mu_x) * var_x ** -1 * (x - mu_x), dim=-1)))

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLz = -0.5 * torch.sum(1 + torch.log(var_z) - mu_z.pow(2) - var_z)

        # To maximize ELBO we minimize loss (-ELBO)
        return -logp + beta*KLz, -logp, beta*KLz


class GMVAE(nn.Module):
    # C: global variable
    # s: local variable
    def __init__(self, channels, dim_z, dim_w, K, arch='beta_vae', device='cpu'):
        super(GMVAE, self).__init__()

        self.dim_z = dim_z
        self.dim_w = dim_w
        self.K = K
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

            self.z_encoder = nn.Sequential(
                View((-1, 256 * 1 * 1)),  # B, 256
                nn.Linear(256, dim_z * 2),  # B, z_dim*2
            )

            self.w_encoder = nn.Sequential(
                View((-1, 256 * 1 * 1)),  # B, 256
                nn.Linear(256, dim_w * 2),  # B, z_dim*2
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
            self.pre_encoder = nn.Sequential(
                View((-1, channels*784)),
                nn.Linear(channels*784, 400),
                nn.ReLU()
            )
            self.z_encoder = nn.Sequential(
                nn.Linear(400, dim_z*2)
            )
            self.w_encoder = nn.Sequential(
                nn.Linear(400, dim_w*2)
            )
            self.decoder = nn.Sequential(
                nn.Linear(dim_z, 400),
                nn.ReLU(),
                nn.Linear(400, channels*784),
                nn.Sigmoid(),
                View((-1, channels, 28, 28))
            )

        self.encoder_d = nn.Sequential(
            # View((-1, dim_beta+dim_w)),  # B, dim_beta+dim_w
            nn.Linear(dim_z+dim_w, 256),  # B, 256
            nn.Tanh(),
            nn.Linear(256, K),  # B, K
            #StableSofmax(L)
            nn.Softmax(dim=1)
        )

        self.z_prior = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_w, 256),  # B, 256
                nn.ReLU(True),
                # nn.Linear(256, 256),  # B, 256
                # nn.ReLU(True),
                nn.Linear(256, dim_z * 2)
            )
            for k in range(K)])


    def _encode(self, x):
        h = self.pre_encoder(x)
        mu_z, std_z = self._z_encode(h)
        mu_w, std_w = self._w_encode(h)

        return mu_z, std_z, mu_w, std_w


    def _z_encode(self, h):
        theta = self.z_encoder(h)
        mu = theta[:, :self.dim_z]
        var = torch.exp(torch.tanh(theta[:, self.dim_z:]))

        return mu, var

    def _w_encode(self, h):
        theta = self.w_encoder(h)
        mu = theta[:, :self.dim_w]
        var = torch.exp(torch.tanh(theta[:, self.dim_w:]))

        return mu, var

    def _encode_d(self, z, w):
        
        input_pi = torch.cat([z, w], dim=1)
        pi = self.encoder_d(input_pi) + 1e-20 # add a constant to avoid log(0)
        return pi


    def reparameterize(self, mu, var):
        std = var**0.5
        eps = torch.randn_like(std)
        return mu + eps*std


    def _z_prior(self, w):
        out = [self.z_prior[k](w) for k in range(self.K)]
        mus_z = torch.stack([out[k][:, :self.dim_z] for k in range(self.K)])
        vars_z = torch.stack([torch.exp(torch.tanh(out[k][:, self.dim_z:])) for k in range(self.K)])
        return mus_z, vars_z


    def _decode(self, z):
        return self.decoder(z)


    def forward(self, x):

        # Encode
        h = self.pre_encoder(x)
        mu_z, var_z = self._z_encode(h)
        z = self.reparameterize(mu_z, var_z)
        mu_w, var_w = self._w_encode(h)
        w = self.reparameterize(mu_w, var_w)
        pi = self._encode_d(z, w)
        mus_z, vars_z = self._z_prior(w)

        # Decode
        mu_x = self._decode(z)

        return mu_x, mu_z, var_z, mu_w, var_w, pi, mus_z, vars_z


    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu_z, var_z, mu_w, var_w, pi, mus_z, vars_z, distribution='gaussian'):

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
        #KLz = -0.5 * torch.sum(1 + torch.log(var_z) - mu_z.pow(2) - var_z)

        KLz = 0
        l=0
        for mu, var in zip(mus_z, vars_z):
            KLz += ( 0.5 * torch.sum( torch.unsqueeze(pi[:, l], 1) * (- 1 + var**-1 * var_z + (mu-mu_z)**2*var**-1
                            +torch.log(var) - torch.log(var_z))))
            l+=1

        KLw = -0.5 * torch.sum(1 + torch.log(var_w) - mu_w.pow(2) - var_w)

        KLd = torch.sum(pi*(torch.log(pi.double()).float()+ torch.log(torch.tensor(self.K, dtype=torch.float).to(self.device)) ))

        return recogn + KLz + KLw + KLd, recogn, KLz, KLw, KLd



class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class Expand(nn.Module):
    def __init__(self, dim, size):
        super(Expand, self).__init__()
        self.dim = dim
        self.size = size

    def forward(self, tensor):
        new_size = list(tensor.shape[:self.dim]) +  self.size + list(tensor.shape[self.dim+1:])
        return tensor.view(new_size)


class Unsqueeze(nn.Module):
    def __init__(self, N):
        super(Unsqueeze, self).__init__()
        self.N = N

    def forward(self, tensor):
        new_size = list(tensor.shape) +  list(torch.ones(self.N))

        return tensor.view(new_size)


class StableSofmax(nn.Module):

    def __init__(self, dims):
        super(StableSofmax, self).__init__()
        self.dims = dims

    def forward(self, x):
        m = torch.max(x, dim=1)[0]
        m = m.repeat(self.dims, 1).t()
        z = x - m
        return torch.softmax(z, dim=1)
