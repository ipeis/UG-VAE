

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

    def encoding(self, model, batch_1, batch_2, folder='./'):

        folder = 'results/' + args.name + '/figs/interpolation/' + folder
        if os.path.isdir(folder) == False:
            os.makedirs(folder)

        # Encode
        h1 = model.pre_encoder(batch_1)
        mu_z1, var_z1 = model._encode_z(h1)
        pi_1 = model._encode_d(mu_z1)
        mu_beta1, var_beta1 = model._encode_beta(h1, pi_1)

        h2 = model.pre_encoder(batch_2)
        mu_z2, var_z2 = model._encode_z(h2)
        pi_2 = model._encode_d(mu_z2)
        mu_beta2, var_beta2 = model._encode_beta(h2, pi_2)

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
        #grid = grid.permute(1, 0, 2, 3, 4)
        #grid = grid.reshape(-1, grid.shape[2], grid.shape[3], grid.shape[4])
        save_image(grid.cpu(),
                   folder + 'interpolation.pdf', nrow=args.steps, padding=1)


    def sampling(self, model, folder='./'):

        folder = 'results/' + args.model_name + '/figs/interpolation/' + folder
        if os.path.isdir(folder) == False:
            os.makedirs(folder)

        # Sample
        # beta1 = torch.randn(dim_beta)
        #d = torch.randint(model.L, [1,1])
        C1 = torch.squeeze(torch.ones(args.dim_C, 1)) * 1.5
        C2 = torch.squeeze(torch.ones(args.dim_C, 1)) * -1.5
        s1 = torch.squeeze(torch.ones(1, args.dim_s)).view(1, args.dim_s) * 5
        s2 = torch.squeeze(torch.ones(1, args.dim_s)).view(1, args.dim_s) * -5

        lambda_ = torch.linspace(0, 1, self.steps)
        C_int = [l * C2 + (1-l) * C1 for l in lambda_]
        s_int = [l * s2 + (1-l) * s1 for l in lambda_]
        grid = []
        for i1 in range(self.steps):
            for i2 in range(self.steps):
                recon = model._decode(s_int[i2], C_int[i1])
                grid.append(recon)
        grid = torch.cat(grid)
        save_image(grid.cpu(),
                   folder + 'sampling_interpolation.pdf', nrow=self.steps, padding=1)




########################################################################################################################
parser = argparse.ArgumentParser(description='Interpolation')
parser.add_argument('--dim_s', type=int, default=10, metavar='N',
                    help='Dimensions for local latent')
parser.add_argument('--dim_C', type=int, default=20, metavar='N',
                    help='Dimensions for global latent')
parser.add_argument('--dataset', type=str, default='celeba',
                    help='Name of the dataset')
parser.add_argument('--arch', type=str, default='beta_vae',
                    help='Architecture for the model')
parser.add_argument('--steps', type=int, default=7, metavar='N',
                    help='Number steps in z variable')
parser.add_argument('--epoch', type=int, default=20,
                    help='Epoch to load')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--model_name', type=str, default='MLVAE/celeba',
                    help='name for the model to be saved')
args = parser.parse_args()


if __name__ == "__main__":

    data_tr, _, data_test = get_data(args.dataset)

    train_loader = torch.utils.data.DataLoader(data_tr, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=True)


    batch_1, _ = iter(train_loader).next()
    batch_2, _ = iter(train_loader).next()

    model = MLVAE(channels=nchannels[args.dataset], dim_s=args.dim_s, dim_C=args.dim_C, arch=args.arch)

    state_dict = torch.load('./results/' + args.model_name + '/checkpoints/checkpoint_' + str(args.epoch) + '.pth',
                            map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)


    #Interpolation(steps=steps).encoding(model, batch_1, batch_2, 'epoch_' + str(epoch) + '/')
    Interpolation(steps=args.steps).sampling(model, 'epoch_' + str(args.epoch) + '/')
    #Interpolation(steps=steps).map_interpolation(model, train_loader, reps=100,  folder='epoch_' + str(epoch) + '/', labels_str=['mnist', 'SVHN'])
