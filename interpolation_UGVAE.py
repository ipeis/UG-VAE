

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
                   folder + 'interpolation.pdf', nrow=steps, padding=1)

    def map_interpolation(self, model, loader, reps = 100, folder='./', labels_str=None):

        # build map
        beta_map=[]
        labels=[]
        for n in range(reps):
            batch, l = iter(loader).next()
            # Encode
            h = model.pre_encoder(batch)
            mu_z, var_z = model._encode_z(h)
            pi = model._encode_d(mu_z)
            mu_beta, var_beta = model._encode_beta(h, pi)
            beta_map.append(mu_beta)
            labels.append(l[0])

        # encode two batches (to interpolate between two samples)
        batch_1, l = iter(loader).next()
        labels.append(l[0])
        batch_2, l = iter(loader).next()
        labels.append(l[0])

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
        #grid = grid.permute(1, 0, 2, 3, 4)
        #grid = grid.reshape(-1, grid.shape[2], grid.shape[3], grid.shape[4])
        save_image(grid.cpu(),
                   folder + 'interpolation.pdf', nrow=steps, padding=1)


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
                ind = labels[:reps]==l
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



    def sampling(self, model, folder='./'):

        folder = 'results/' + name + '/figs/interpolation/' + folder
        if os.path.isdir(folder) == False:
            os.makedirs(folder)

        for d in range(model.L):
            print('Component ' + str(d))
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
                mu_z = torch.squeeze(mus_z[d])
                var_z = torch.squeeze(vars_z[d])
                z1 = torch.squeeze(torch.ones_like(mu_z)) * mu_z - 3
                z2 = torch.squeeze(torch.ones_like(mu_z)) * mu_z + 3
                z = torch.stack([l * z2 + (1 - l) * z1 for l in lambda_])
                """
                z = torch.stack(
                    [Normal(mu_z, torch.diag(var_z)).sample() for i in range(5)])
                z = torch.stack(
                    [mu_z for i in range(5)])
                """
                recon = model._decode(z, global_int[s1])
                grid.append(recon)
            grid = torch.cat(grid)
            #grid = grid.permute(1, 0, 2, 3, 4)
            #grid = grid.reshape(-1, grid.shape[2], grid.shape[3], grid.shape[4])
            save_image(grid.cpu(),
                       folder + 'sampling_interpolation_' + str(d) + '.pdf', nrow=self.steps, padding=1)




########################################################################################################################
parser = argparse.ArgumentParser(description='Interpolation')
parser.add_argument('--dim_z', type=int, default=20, metavar='N',
                    help='Dimensions for local latent')
parser.add_argument('--dim_beta', type=int, default=50, metavar='N',
                    help='Dimensions for global latent')
parser.add_argument('--L', type=int, default=20, metavar='N',
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
L = args.L

if __name__ == "__main__":

    data_tr, _, data_test = get_data(dataset)

    train_loader = torch.utils.data.DataLoader(data_tr, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=True)


    batch_1, _ = iter(train_loader).next()
    batch_2, _ = iter(train_loader).next()

    model = UGVAE(channels=nchannels[args.dataset], dim_z=dim_z, dim_beta=dim_beta, L=L, arch=args.arch)
    state_dict = torch.load('./results/' + name + '/checkpoints/checkpoint_' + str(epoch) + '.pth',
                            map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)


    Interpolation(steps=steps).encoding(model, batch_1, batch_2, 'epoch_' + str(epoch) + '/')
    Interpolation(steps=steps).sampling(model, 'epoch_' + str(epoch) + '/')
    #Interpolation(steps=steps).map_interpolation(model, train_loader, reps=100,  folder='epoch_' + str(epoch) + '/', labels_str=['mnist', 'SVHN'])
