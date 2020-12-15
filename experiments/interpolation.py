import sys
sys.path.append('..')
from models import *
import argparse
from datasets import *
from torchvision.utils import save_image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

#----------------------------------------------------------------------------------------------------------------------#
# Arguments
parser = argparse.ArgumentParser(description='Interpolation in UG-VAE')
parser.add_argument('--dim_z', type=int, default=20, metavar='N',
                    help='Dimensions for local latent z (default: 20)')
parser.add_argument('--dim_beta', type=int, default=50, metavar='N',
                    help='Dimensions for global latent beta (default: 20)')
parser.add_argument('--K', type=int, default=20, metavar='N',
                    help='Number of components for the Gaussian mixture (default: 20)')
parser.add_argument('--var_x', type=float, default=2e-1, metavar='N',
                    help='Variance of p(x|z,beta) (default: 2e-1)')
parser.add_argument('--dataset', type=str, default='celeba',
                    help='Name of the dataset (default: celeba)')
parser.add_argument('--arch', type=str, default='beta_vae',
                    help='Architecture for the model (default: beta_vae)')
parser.add_argument('--steps', type=int, default=7, metavar='N',
                    help='Number steps in local space and global space (default: 7)')
parser.add_argument('--epoch', type=int, default=10,
                    help='Epoch to load (default: 10)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--model_name', type=str, default='celeba',
                    help='name for the model to load and save figs (default: celeba)')
args = parser.parse_args()



#----------------------------------------------------------------------------------------------------------------------#
class Interpolation():

    def __init__(self, steps):
        self.steps = steps


    #------------------------------------------------------------------------------------------------------------------#
    def encoding(self, model, loader, folder='./'):
        """
        Local and global interpolation from q(z1, beta1 | batch1) to q(z2, beta2 | batch2)
        z1 is obtained only for first image in each batch
        Args:
            model: UGVAE model
            batch_1: first batch
            batch_2: second batch
            folder: folder to save figures

        Returns:

        """
        # Create subfolder in log dir
        folder = '../results/' + args.model_name + '/figs/interpolation/' + folder
        if os.path.isdir(folder) == False:
            os.makedirs(folder)

        # Encode two batches (to interpolate between two samples)
        batch_1, l = iter(loader).next()
        batch_2, l = iter(loader).next()

        # Encode first batch
        h1 = model.pre_encoder(batch_1)
        mu_z1, var_z1 = model._encode_z(h1)
        pi_1 = model._encode_d(mu_z1)
        mu_beta1, var_beta1 = model._encode_beta(h1, pi_1)

        # Encode second batch
        h2 = model.pre_encoder(batch_2)
        mu_z2, var_z2 = model._encode_z(h2)
        pi_2 = model._encode_d(mu_z2)
        mu_beta2, var_beta2 = model._encode_beta(h2, pi_2)

        # Take local encode of first image in each batch
        mu_z1 = mu_z1[0]
        mu_z2 = mu_z2[0].view(-1, mu_z2.shape[-1])

        # Interpolation
        lambda_ = torch.linspace(0, 1, self.steps)
        local_int = [l * mu_z2 + (1-l) * mu_z1 for l in lambda_]
        global_int = [l * mu_beta2 + (1-l) * mu_beta1 for l in lambda_]
        grid = []
        for s1 in range(self.steps):
            for s2 in range(self.steps):
                recon = model._decode(local_int[s2], global_int[s1])
                grid.append(recon)
        grid = torch.cat(grid)
        save_image(grid.cpu(),
                   folder + 'interpolation.pdf', nrow=args.steps, padding=1)


    #------------------------------------------------------------------------------------------------------------------#
    def map_interpolation(self, model, loader, reps = 100, folder='./', labels_str=None):
        """
        Local and global interpolation from q(z1, beta1 | batch1) to q(z2, beta2 | batch2)
        z1 is obtained only for first image in each batch
        This function also builds a tSNE map showing the global space
        Args:
            model: UGVAE model
            loader: to extract batches
            reps: number of batches to build the global map
            folder: folder to save figures
            labels_str: list with legend for the map ['fist image', 'second_image']

        Returns:

        """
        # Create subfolder in log dir
        folder = '../results/' + args.model_name + '/figs/interpolation/' + folder
        if os.path.isdir(folder) == False:
            os.makedirs(folder)

        # Obtain global encodes for the map
        beta_map=[]
        labels=[]
        for n in range(reps):
            # Load batch
            batch, l = iter(loader).next()
            # Encode
            h = model.pre_encoder(batch)
            mu_z, var_z = model._encode_z(h)
            pi = model._encode_d(mu_z)
            mu_beta, var_beta = model._encode_beta(h, pi)
            beta_map.append(mu_beta)
            labels.append(l[0])

        # Encode two batches (to interpolate between two samples)
        batch_1, l = iter(loader).next()
        labels.append(l[0])
        batch_2, l = iter(loader).next()
        labels.append(l[0])

        # Encode first batch
        h1 = model.pre_encoder(batch_1)
        mu_z1, var_z1 = model._encode_z(h1)
        pi_1 = model._encode_d(mu_z1)
        mu_beta1, var_beta1 = model._encode_beta(h1, pi_1)

        # Encode second batch
        h2 = model.pre_encoder(batch_2)
        mu_z2, var_z2 = model._encode_z(h2)
        pi_2 = model._encode_d(mu_z2)
        mu_beta2, var_beta2 = model._encode_beta(h2, pi_2)

        # Take local encode of first image in each batch
        mu_z1 = mu_z1[0]
        mu_z2 = mu_z2[0].view(-1, mu_z2.shape[-1])

        # Interpolation
        lambda_ = torch.linspace(0, 1, self.steps)
        local_int = [l * mu_z2 + (1-l) * mu_z1 for l in lambda_]
        global_int = [l * mu_beta2 + (1-l) * mu_beta1 for l in lambda_]
        grid = []
        for s1 in range(self.steps):
            for s2 in range(self.steps):
                recon = model._decode(local_int[s2], global_int[s1])
                grid.append(recon)
        grid = torch.cat(grid)
        save_image(grid.cpu(),
                   folder + 'interpolation.pdf', nrow=args.steps, padding=1)

        # MAP
        beta_map += global_int
        beta_map = torch.stack(beta_map)

        print('Training t-SNE...')
        beta_tsne = TSNE(n_components=2).fit_transform(beta_map.detach().numpy())

        plt.figure(figsize=(8, 8))

        if labels_str != None:
            labels = np.array(labels)
            print(labels.shape)
            for l in range(len(np.unique(labels))):
                ind = labels[:reps] == l
                plt.plot(beta_tsne[:reps][ind, 0], beta_tsne[:reps][ind, 1], 'o', label=labels_str[l])
            plt.legend(loc='best')
            plt.savefig(folder + 'interpolation_map.pdf')
        else:

            plt.plot(beta_tsne[:, 0], beta_tsne[:, 1], 'o')
        plt.plot(beta_tsne[-self.steps, 0], beta_tsne[-self.steps, 1], 'ko')
        plt.plot(beta_tsne[-1, 0], beta_tsne[-1, 1], 'k>')
        plt.plot(beta_tsne[-self.steps:, 0], beta_tsne[-self.steps:, 1], 'k-o', label='Interpolation')
        plt.legend(loc='best')
        plt.savefig(folder + 'interpolation_map.pdf')


    #------------------------------------------------------------------------------------------------------------------#
    def sampling(self, model, folder='./'):
        """
        Local and global interpolation from PRIORS p(z|d, beta) and p(beta)
        For fixed d, we move diagonally trough the Gaussian probability masses
        For beta, interpolation goes linearly from beta1=[-1.5, -1.5, ..., -1.5] to beta2=[1.5, 1.5, ..., 1.5]
        As beta determines the prior of z, p(z|d, beta):
        For z, interpolation goes linearly from z1=[muz-3, muz-3, ..., muz-3] to z2=[muz+3, muz+3, ..., muz+3]
        Args:
            model: UGVAE model
            folder: folder to save figures

        Returns:

        """
        # Create subfolder in log dir
        folder = '../results/' + args.model_name + '/figs/interpolation/' + folder
        if os.path.isdir(folder) == False:
            os.makedirs(folder)

        # For each component d
        for d in range(model.K):
            print('Component ' + str(d))
            # Initial and final beta
            beta1 = torch.squeeze(torch.ones(model.dim_beta, 1)) * 1.5
            beta2 = torch.squeeze(torch.ones(model.dim_beta, 1)) * -1.5

            # Interpolation
            lambda_ = torch.linspace(0, 1, self.steps)
            global_int = [l * beta2 + (1-l) * beta1 for l in lambda_]
            grid = []
            # For each beta, determine p(z|d, beta), interpolate in z domain and decode
            for s1 in range(self.steps):
                mus_z, vars_z = model._z_prior(global_int[s1])
                mu_z = torch.squeeze(mus_z[d])
                z1 = torch.squeeze(torch.ones_like(mu_z)) * mu_z - 3
                z2 = torch.squeeze(torch.ones_like(mu_z)) * mu_z + 3
                z = torch.stack([l * z2 + (1 - l) * z1 for l in lambda_])
                recon = model._decode(z, global_int[s1])
                grid.append(recon)
            # Save grid with local and global interpolations
            grid = torch.cat(grid)
            save_image(grid.cpu(),
                       folder + 'sampling_interpolation_' + str(d) + '.pdf', nrow=self.steps, padding=1)



#----------------------------------------------------------------------------------------------------------------------#
# Main
if __name__ == "__main__":

    # Load data
    data, _, _ = get_data(args.dataset, path='../data/')
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True)

    # Load model
    model = UGVAE(channels=nchannels[args.dataset], dim_z=args.dim_z, dim_beta=args.dim_beta, K=args.K, arch=args.arch)
    state_dict = torch.load('../results/' + args.model_name + '/checkpoints/checkpoint_' + str(args.epoch) + '.pth',
                            map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    # Perform the interpolation
    Interpolation(steps=args.steps).sampling(model, 'epoch_' + str(args.epoch) + '/')