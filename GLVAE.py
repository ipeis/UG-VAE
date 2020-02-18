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



    def train_epoch(self, epoch, train_loader, optimizer, cuda=False, log_interval=10):

        cuda = cuda and torch.cuda.is_available()
        device = torch.device("cuda" if cuda else "cpu")
        self.train()
        train_loss = 0
        train_rec = 0
        train_kl_l = 0
        train_kl_g = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, var, mu_g, var_g = self.forward(data)
            loss, rec, kl_l, kl_g = loss_function(recon_batch, data, mu, var, mu_g, var_g)
            loss.backward()
            train_loss += loss.item()
            train_rec += rec.item()
            train_kl_l += kl_l.item()
            train_kl_g += kl_g.item()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))

        return train_loss / len(train_loader.dataset), train_rec / len(train_loader.dataset), \
               train_kl_l / len(train_loader.dataset), train_kl_g / len(train_loader.dataset)


    def test(self, epoch, test_loader, cuda=False, model_name='model'):
        cuda = cuda and torch.cuda.is_available()
        device = torch.device("cuda" if cuda else "cpu")
        self.eval()
        test_loss = 0
        test_rec = 0
        test_kl_l = 0
        test_kl_g = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                data = data.to(device)
                recon_batch, mu, var, mu_g, var_g = self.forward(data)
                loss, rec, kl_l, kl_g = loss_function(recon_batch, data, mu, var, mu_g, var_g)
                test_loss += loss.item()
                test_rec += rec.item()
                test_kl_l += kl_l.item()
                test_kl_g += kl_g.item()

                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n],
                                          recon_batch.view(test_loader.batch_size, 1, 28, 28)[:n]])
                    save_image(comparison.cpu(),
                             'results/' + model_name + '/figs/reconstruction_' + str(epoch) + '.png', nrow=n)

        test_loss /= len(test_loader.dataset)
        test_rec /= len(test_loader.dataset)
        test_kl_l /= len(test_loader.dataset)
        test_kl_g /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

        return test_loss, test_rec, test_kl_l, test_kl_g


    def plot_latent(self, epoch, test_loader, cuda=False, model_name='model'):
        cuda = cuda and torch.cuda.is_available()
        device = torch.device("cuda" if cuda else "cpu")
        self.eval()
        test_loss = 0
        z = []
        labels = []
        with torch.no_grad():
            for i, (data, l) in enumerate(test_loader):
                data = data.to(device)
                labels.append(l.to(device))
                recon_batch, mu, var, mu_g, var_g = self.forward(data)
                z.append(self.reparameterize(mu, var))

        z = torch.cat(z).cpu().numpy()[:1000]
        labels = torch.cat(labels).cpu().numpy()[:1000]

        #print('Performing t-SNE...')

        #X = TSNE(n_components=2, random_state=0).fit_transform(z)

        X = z
        plt.close('all')
        colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple'
        for i, c in enumerate(colors):
            plt.scatter(X[labels == i, 0], X[labels == i, 1], c=c, label=str(i))

        plt.legend(loc='best')
        plt.savefig('results/' + model_name + '/figs/local_latent_epoch_' + str(epoch) + '.pdf')


    def plot_global_latent(self, epoch, nsamples, nreps, nims, cuda=False, model_name='model'):
        cuda = cuda and torch.cuda.is_available()
        device = torch.device("cuda" if cuda else "cpu")
        self.eval()

        sample_z = torch.randn(nsamples, self.dim_z).to(device)

        #mu_Z, var_Z = model.global_encode(sample_z)
        #samples_Z = torch.stack([model.reparameterize(mu_Z, var_Z) for n in range(nreps)])
        samples_Z = torch.randn(nreps, self.dim_Z).to(device)

        pick = np.random.randint(0, len(samples_Z), nims)
        for s, sample_Z in enumerate(samples_Z[pick]):
            sample = self.decode(sample_z, sample_Z).cpu()
            sample = sample.view(nsamples, 1, 28, 28)
            sample = make_grid(sample, nrow=2, padding=2)
            save_image(sample, 'results/' + model_name + '/figs/Z' + str(s) + '_' + str(epoch) + '.png')

        Z = samples_Z.detach().cpu().numpy()

        print('Performing t-SNE...')
        X = TSNE(n_components=2, random_state=0).fit_transform(Z)

        paths = ['results/' + model_name + '/figs/Z' + str(s) + '_' + str(epoch) + '.png' for s in range(nreps)]

        fig, ax = plt.subplots(figsize=(12, 12))
        plt.scatter(X[:, 0], X[:, 1])

        for x0, y0, path in zip(X[pick, 0], X[pick, 1], paths):
            ab = AnnotationBbox(OffsetImage(plt.imread(path)), (x0, y0), frameon=False)
            ax.add_artist(ab)

        plt.savefig('results/' + model_name + '/figs/global_latent_epoch_' + str(epoch) + '.pdf')

    def plot_losses(self, tr_losses, test_losses, tr_recs, test_recs, tr_kl_ls, test_kl_ls, tr_kl_gs, test_kl_gs,
                    model_name='model'):
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        plt.figure()
        plt.semilogy(tr_losses, color=colors[0], label='train_loss')
        plt.semilogy(test_losses, color=colors[0], linestyle=':')
        plt.semilogy(tr_recs, color=colors[1], label='train_rec')
        plt.semilogy(test_recs, color=colors[1], linestyle=':')
        plt.semilogy(tr_kl_ls, color=colors[2], label='KL_loc')
        plt.semilogy(test_kl_ls, color=colors[2], linestyle=':')
        plt.semilogy(tr_kl_gs, color=colors[3], label='KL_glob')
        plt.semilogy(test_kl_gs, color=colors[3], linestyle=':')
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('mean loss')
        plt.legend(loc='best')
        plt.savefig('results/' + model_name + '/figs/losses.pdf')

    def save_model(self, epoch, model_name='model'):

        folder = 'results/' + model_name + '/checkpoints/'
        if os.path.isdir(folder) == False:
            os.makedirs(folder)

        torch.save(self.state_dict(), folder + '/checkpoint_' + str(epoch) + '.pth')


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

