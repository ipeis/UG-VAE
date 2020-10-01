

from models import *
import argparse
from torchvision import datasets, transforms
from torch import nn, optim
from datasets import *
from torchvision.utils import save_image
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE




class Interpolation():

    def __init__(self, steps):
        self.steps = steps

    def sampling(self, model, folder='./'):

        folder = 'results/' + name + '/figs/interpolation/' + folder
        if os.path.isdir(folder) == False:
            os.makedirs(folder)

        for k in range(model.K):
            print('Component ' + str(k))
            # Sample
            # beta1 = torch.randn(dim_beta)
            #d = torch.randint(model.L, [1,1])
            beta1 = torch.squeeze(torch.ones(dim_beta, 1)) * 1.5
            beta2 = torch.squeeze(torch.ones(dim_beta, 1)) * -1.5

            lambda_ = torch.linspace(0, 1, self.steps)
            global_int = [l * beta2 + (1-l) * beta1 for l in lambda_]
            grid = []
            for s1 in range(self.steps):
                mus_z, vars_z = model._z_prior(global_int[s1])
                mu_z = torch.squeeze(mus_z[k])
                var_z = torch.squeeze(vars_z[k])
                z1 = torch.squeeze(torch.ones_like(mu_z)) * mu_z - 3
                z2 = torch.squeeze(torch.ones_like(mu_z)) * mu_z + 3
                z = torch.stack([l * z2 + (1 - l) * z1 for l in lambda_])

                recon = model._decode(z, global_int[s1])
                grid.append(recon)
            grid = torch.cat(grid)
            save_image(grid.cpu(),
                       folder + 'sampling_interpolation_' + str(k) + '.pdf', nrow=self.steps, padding=1)




########################################################################################################################
parser = argparse.ArgumentParser(description='Interpolation')
parser.add_argument('--dim_z', type=int, default=20, metavar='N',
                    help='Dimensions for local latent')
parser.add_argument('--dim_beta', type=int, default=50, metavar='N',
                    help='Dimensions for global latent')
parser.add_argument('--K', type=int, default=20, metavar='N',
                    help='Number of components for the Gaussian Global mixture')
parser.add_argument('--var_x', type=float, default=2e-1, metavar='N',
                    help='Number of components for the Gaussian Global mixture')
parser.add_argument('--dataset', type=str, default='celeba',
                    help='Name of the dataset')
parser.add_argument('--arch', type=str, default='beta_vae',
                    help='Architecture for the model')
parser.add_argument('--steps', type=int, default=7, metavar='N',
                    help='Number steps in z variable')
parser.add_argument('--epoch', type=int, default=17,
                    help='Epoch to load')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--model_name', type=str, default='UG-VAE/celeba_2000',
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

    data_tr, _, data_test = get_data(dataset)

    train_loader = torch.utils.data.DataLoader(data_tr, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=True)


    batch_1, _ = iter(train_loader).next()
    batch_2, _ = iter(train_loader).next()

    model = UGVAE(channels=nchannels[args.dataset], dim_z=dim_z, dim_beta=dim_beta, K=K, arch=args.arch)
    state_dict = torch.load('./results/' + name + '/checkpoints/checkpoint_' + str(epoch) + '.pth',
                            map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    Interpolation(steps=steps).sampling(model, 'epoch_' + str(epoch) + '/')
