

from models import *
import argparse
from torchvision import datasets, transforms
from torch import nn, optim
from datasets import *
from torchvision.utils import save_image
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn import mixture



class Interpolation():

    def __init__(self, steps):
        self.steps = steps

    def map_interpolation(self, model, loader, loader_mix, reps = 100, folder='./', labels_str=None):

        # mixed data
        beta_map = []
        labels = []
        for n in range(reps):
            batch, _ = iter(loader_mix).next()
            # Encode
            h = model.pre_encoder(batch)
            mu_z, var_z = model._encode_z(h)
            pi = model._encode_d(mu_z)
            mu_beta, var_beta = model._encode_beta(h, pi)
            beta_map.append(mu_beta)
            labels.append(2)

        for n in range(reps):
            batch, l = iter(loader).next()
            # Encode
            h = model.pre_encoder(batch)
            mu_z, var_z = model._encode_z(h)
            pi = model._encode_d(mu_z)
            mu_beta, var_beta = model._encode_beta(h, pi)
            beta_map.append(mu_beta)
            labels.append(l[0])

        folder = 'results/' + name + '/figs/interpolation/' + folder
        if os.path.isdir(folder) == False:
            os.makedirs(folder)

        # encode two batches (to interpolate between two samples)
        batch_1, l = iter(loader).next()
        save_image(batch_1.squeeze()[:5], folder + 'batch_1.pdf', nrow=5)
        labels.append(l[0])
        batch_2, l = iter(loader).next()
        save_image(batch_2.squeeze()[:5], folder + 'batch_2.pdf', nrow=5)
        labels.append(l[0])


        # Encode
        h1 = model.pre_encoder(batch_1)
        mu_z1, var_z1 = model._encode_z(h1)
        pi_1 = model._encode_d(mu_z1)
        mu_beta1, var_beta1 = model._encode_beta(h1, pi_1)

        h2 = model.pre_encoder(batch_2)
        mu_z2, var_z2 = model._encode_z(h2)
        pi_2 = model._encode_d(mu_z2)
        mu_beta2, var_beta2 = model._encode_beta(h2, pi_2)

        # INTERPOLATION
        mu_z1 = mu_z1[0]
        mu_z2 = mu_z2[0].view(-1, mu_z2.shape[-1])

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
                   folder + 'interpolation.pdf', nrow=steps, padding=1)

        # MAP
        beta_map += global_int
        beta_map = torch.stack(beta_map)

        print('Training t-SNE...')
        beta_tsne = TSNE(n_components=2).fit_transform(beta_map.detach().numpy())

        plt.figure(figsize=(6, 6))

        markers = ['s', '^']
        colors = [(52/256, 77/256, 155/256), (53/256, 160/256, 38/256)]

        if labels_str != None:
            labels = np.array(labels)
            ind = labels[:reps] == 2
            plt.plot(beta_tsne[:reps][ind, 0], beta_tsne[:reps][ind, 1], '.', color='0.3', label=labels_str[2])
            for l in range(len(np.unique(labels)[:-1])):
                ind = labels[reps:2*reps]==l
                plt.plot(beta_tsne[reps:2*reps][ind, 0], beta_tsne[reps:2*reps][ind, 1], 'o', color=colors[l], marker=markers[l], alpha = 0.7, label=labels_str[l])
        else:
            plt.plot(beta_tsne[:, 0], beta_tsne[:, 1], 'o')
        plt.plot(beta_tsne[-self.steps, 0], beta_tsne[-self.steps, 1], 'ko')
        plt.plot(beta_tsne[-1, 0], beta_tsne[-1, 1], 'k>')
        plt.plot(beta_tsne[-self.steps:, 0], beta_tsne[-self.steps:, 1], 'k-o', label='Interpolation')
        plt.legend(loc='best', fontsize=12)
        plt.grid()
        plt.savefig(folder + 'interpolation_map.pdf')




########################################################################################################################
parser = argparse.ArgumentParser(description='Interpolation')
parser.add_argument('--dim_z', type=int, default=40, metavar='N',
                    help='Dimensions for local latent')
parser.add_argument('--dim_beta', type=int, default=40, metavar='N',
                    help='Dimensions for global latent')
parser.add_argument('--K', type=int, default=40, metavar='N',
                    help='Number of components for the Gaussian Global mixture')
parser.add_argument('--var_x', type=float, default=2e-1, metavar='N',
                    help='Number of components for the Gaussian Global mixture')
parser.add_argument('--dataset', type=str, default='celeba_faces_batch',
                    help='Name of the dataset')
parser.add_argument('--arch', type=str, default='beta_vae',
                    help='Architecture for the model')
parser.add_argument('--steps', type=int, default=7, metavar='N',
                    help='Number steps in z variable')
parser.add_argument('--epoch', type=int, default=10,
                    help='Epoch to load')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--model_name', type=str, default='UG-VAE/celeba_faces',
                    help='name for the model to be saved')
args = parser.parse_args()


name = args.model_name
epoch = args.epoch
steps = args.steps
dataset = args.dataset
dim_z = args.dim_z
dim_beta = args.dim_beta
K = args.K

if __name__ == "__main__":

    data, _, _ = get_data(dataset)
    data_mix, _, _ = get_data(dataset[:-6])

    loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True)
    loader_mix = torch.utils.data.DataLoader(data_mix, batch_size=args.batch_size, shuffle=True)


    model = UGVAE(channels=nchannels[args.dataset], dim_z=dim_z, dim_beta=dim_beta, K=K, arch=args.arch)
    state_dict = torch.load('./results/' + name + '/checkpoints/checkpoint_' + str(epoch) + '.pth',
                            map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    Interpolation(steps=steps).map_interpolation(model, loader, loader_mix, reps=100,  folder='epoch_' + str(epoch) + '/', labels_str=['celebA', 'FACES', 'mix'])
