

from models import *
import argparse
from torchvision import datasets, transforms
from torch import nn, optim
from datasets import *
from torchvision.utils import save_image



class Interpolation():

    def __init__(self, steps):
        self.steps = steps

    def build(self, model, batch_1, batch_2, folder='./'):

        folder = 'results/' + name + '/figs/interpolation/' + folder
        if os.path.isdir(folder) == False:
            os.makedirs(folder)

        mu_z1, var_z1, p1, mus_beta, vars_beta = model._encode(batch_1)
        mu_z2, var_z2, p2, mus_beta, vars_beta = model._encode(batch_2)

        mu_z1 = mu_z1[0]
        mu_z2 = mu_z2[0].view(-1, mu_z2.shape[-1])
        p1_ = torch.argmax(p1).detach().numpy()
        mu_beta1 = mus_beta[p1_]
        p2_ = torch.argmax(p2).detach().numpy()
        mu_beta2 = mus_beta[p2_]

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
                   folder + 'interpolation.png', nrow=steps, padding=1)

        # Interpolation in K
        grid = []
        for s1 in range(self.steps):
            for k2 in range(model.K):
                recon = model._decode(local_int[s1], mus_beta[k2])
                grid.append(recon)
        grid = torch.cat(grid)
        # grid = grid.permute(1, 0, 2, 3, 4)
        # grid = grid.reshape(-1, grid.shape[2], grid.shape[3], grid.shape[4])
        save_image(grid.cpu(),
                   folder + 'interpolation_K.png', nrow=model.K, padding=1)


########################################################################################################################
parser = argparse.ArgumentParser(description='Interpolation')
parser.add_argument('--dim_z', type=int, default=10, metavar='N',
                    help='Dimensions for local latent')
parser.add_argument('--dim_beta', type=int, default=10, metavar='N',
                    help='Dimensions for global latent')
parser.add_argument('--K', type=int, default=10, metavar='N',
                    help='Number of components for the Gaussian Global mixture')
parser.add_argument('--dataset', type=str, default='celeba',
                    help='Name of the dataset')
parser.add_argument('--arch', type=str, default='beta_vae',
                    help='Architecture for the model')
parser.add_argument('--steps', type=int, default=4, metavar='N',
                    help='Number steps in z variable')
parser.add_argument('--epoch', type=int, default=10,
                    help='Epoch to load')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--model_name', type=str, default='trained',
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

    model = MGLVAEp(channels=nchannels[args.dataset], dim_z=dim_z, dim_beta=dim_beta, K=K, arch=args.arch)
    state_dict = torch.load('./results/' + name + '/checkpoints/checkpoint_' + str(epoch) + '.pth',
                            map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)


    Interpolation(steps=steps).build(model, batch_1, batch_2, 'epoch_' + str(epoch) + '/')
