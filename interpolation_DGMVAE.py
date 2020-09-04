

from models import *
import argparse
from torchvision import datasets, transforms
from torch import nn, optim
from datasets import *
from torchvision.utils import save_image



class Interpolation():

    def __init__(self, steps):
        self.steps = steps

    def encoding(self, model, batch_1, batch_2, folder='./'):

        folder = 'results/' + name + '/figs/interpolation/' + folder
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
                   folder + 'interpolation.png', nrow=steps, padding=1)

    def sampling(self, model, folder='./'):

        folder = 'results/' + name + '/figs/interpolation/' + folder
        if os.path.isdir(folder) == False:
            os.makedirs(folder)

        # Sample
        # beta1 = torch.randn(dim_beta)
        #d = torch.randint(model.L, [1,1])
        w1 = torch.squeeze(torch.ones(dim_w, 1)) * 3
        w2 = torch.squeeze(torch.ones(dim_w, 1)) * -3

        lambda_ = torch.linspace(0, 1, self.steps)
        global_w = [l * w2 + (1-l) * w1 for l in lambda_]

        for l in range(model.L):
            print('Component ' + str(l))
            grid = []
            for k in range(model.K):
                for s in range(self.steps):
                    beta = model._beta_mix(global_w[s])[0][k] # mean
                    mu_z = model._z_mix(beta)[0][l]
                    var_z = model._z_mix(beta)[1][l]
                    #z = torch.stack(
                    #    [Normal(mu_z, torch.diag(var_z)).sample() for i in range(10)])
                    recon = model._decode(mu_z.unsqueeze(0), beta)
                    grid.append(recon)
            grid = torch.cat(grid)
            save_image(grid.cpu(),
                       folder + 'sampling_interpolation_L' + str(l) + '.png', nrow=steps, padding=1)

        #grid = grid.permute(1, 0, 2, 3, 4)
        #grid = grid.reshape(-1, grid.shape[2], grid.shape[3], grid.shape[4])





########################################################################################################################
parser = argparse.ArgumentParser(description='Interpolation')
parser.add_argument('--dim_z', type=int, default=10, metavar='N',
                    help='Dimensions for local latent')
parser.add_argument('--L', type=int, default=10, metavar='N',
                    help='Number of components for the Gaussian Local mixture')
parser.add_argument('--dim_beta', type=int, default=10, metavar='N',
                    help='Dimensions for global latent')
parser.add_argument('--K', type=int, default=5, metavar='N',
                    help='Number of components for the Gaussian Global mixture')
parser.add_argument('--dim_w', type=int, default=20, metavar='N',
                    help='Dimensions for global noise')
parser.add_argument('--var_x', type=float, default=2e-1, metavar='N',
                    help='Number of components for the Gaussian Global mixture')
parser.add_argument('--dataset', type=str, default='mnist',
                    help='Name of the dataset')
parser.add_argument('--arch', type=str, default='k_vae',
                    help='Architecture for the model')
parser.add_argument('--steps', type=int, default=10, metavar='N',
                    help='Number steps in z variable')
parser.add_argument('--epoch', type=int, default=12,
                    help='Epoch to load')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--model_name', type=str, default='DGMVAE/mnist_series',
                    help='name for the model to be saved')
args = parser.parse_args()


name = args.model_name
epoch = args.epoch
steps = args.steps
dataset = args.dataset
dim_z = args.dim_z
L = args.L
dim_beta = args.dim_beta
K = args.K
dim_w = args.dim_w
var_x = args.var_x

if __name__ == "__main__":

    data_tr, _, data_test = get_data(dataset)

    train_loader = torch.utils.data.DataLoader(data_tr, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=True)


    batch_1, _ = iter(train_loader).next()
    batch_2, _ = iter(train_loader).next()

    model = DGMVAE(channels=nchannels[args.dataset], dim_z=dim_z, L=L, dim_beta=dim_beta, K=K, dim_w=dim_w, var_x=var_x, arch=args.arch)
    state_dict = torch.load('./results/' + name + '/checkpoints/checkpoint_' + str(epoch) + '.pth',
                            map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)


    #Interpolation(steps=steps).encoding(model, batch_1, batch_2, 'epoch_' + str(epoch) + '/')
    Interpolation(steps=steps).sampling(model, 'epoch_' + str(epoch) + '/')
