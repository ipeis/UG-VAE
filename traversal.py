

from GLVAE import *
from MLVAE import *
import argparse
from torchvision import datasets, transforms
from torch import nn, optim
from datasets import *
from torchvision.utils import save_image



class Traversal():

    def __init__(self, zmax, steps):
        self.zmax = zmax
        self.steps = steps

    def build(self, model, data, folder='./'):

        folder = 'results/' + name + '/figs/traversal/' + folder
        if os.path.isdir(folder) == False:
            os.makedirs(folder)

        zt = torch.linspace(-self.zmax, self.zmax, self.steps)
        if args.model=='mlvae':
            z, _, zg, _ = model._encode(data)
        else:
            z, _ = model._encode(data)

        for dim in range(z.shape[-1]):
            z_copy = z.clone().detach()
            grid = []
            for step in zt:
                z_copy[:, dim] = step
                if args.model == 'glvae':
                    zg, _ = model._global_encode(z_copy)
                recon_data = model._decode (z_copy, zg)
                grid.append(recon_data)
            grid = torch.stack(grid)
            grid = grid.permute(1, 0, 2, 3, 4)
            grid = grid.reshape(-1, grid.shape[2], grid.shape[3], grid.shape[4])
            save_image(grid.cpu(),
                       folder + 'z' + str(dim) + '.png', nrow=steps)


class GlobalTraversal():
    def __init__(self, Zmax, steps):
        self.Zmax = Zmax
        self.steps = steps

    def build(self, model, data, folder='./'):

        folder = 'results/' + name + '/figs/traversal/' + folder
        if os.path.isdir(folder) == False:
            os.makedirs(folder)

        batch_grid_size = int(np.sqrt(len(batch)))

        Zt = torch.linspace(-self.Zmax, self.Zmax, self.steps)
        if args.model=='mlvae':
            z, _, zg, _ = model._encode(data)
        elif args.model=='glvae':
            z, _ = model._encode(data)
            zg, _ = model._global_encode(z)
        for dim in range(z.shape[-1]):
            zg_copy = zg.clone().detach()
            grid = []
            for step in Zt:
                zg_copy[dim] = step
                recon_data = model._decode (z, zg_copy)#.view(1, 3, batch_grid_size*64, batch_grid_size*64)
                grid.append(recon_data)
            grid = torch.stack(grid)
            #grid = grid.permute(1, 0, 2, 3, 4)
            grid = grid.reshape(-1, grid.shape[2], grid.shape[3], grid.shape[4])
            save_image(grid.cpu(),
                       folder + 'zg' + str(dim) + '.png', nrow=batch_size)



########################################################################################################################
parser = argparse.ArgumentParser(description='Plot q(z|x)')
parser.add_argument('--dim_z', type=int, default=10, metavar='N',
                    help='Dimensions for local latent')
parser.add_argument('--dim_Z', type=int, default=10, metavar='N',
                    help='Dimensions for global latent')
parser.add_argument('--dataset', type=str, default='celeba',
                    help='Name of the dataset')
parser.add_argument('--arch', type=str, default='beta_vae',
                    help='Architecture for the model')
parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--steps', type=int, default=4, metavar='N',
                    help='Number steps in z variable')
parser.add_argument('--zmax', type=float, default=3, metavar='N',
                    help='[-zmax, zmax] for z range')
parser.add_argument('--epoch', type=int, default=10,
                    help='Epoch to load')
parser.add_argument('--model', type=str, default='glvae',
                    help='model design (glvae or mlvae)')
parser.add_argument('--model_name', type=str, default='trained',
                    help='name for the model to be saved')
args = parser.parse_args()


name = args.model_name
epoch = args.epoch
batch_size = args.batch_size
steps = args.steps
zmax = args.zmax
dataset = args.dataset
dim_z = args.dim_z
dim_Z = args.dim_Z


if __name__ == "__main__":

    data_tr, _, data_test = get_data(dataset)

    train_loader = torch.utils.data.DataLoader(data_tr, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True)


    batch, _ = iter(train_loader).next()

    if args.model == 'mlvae':
        model = MLVAE(channels=nchannels[args.dataset], dim_z=dim_z, dim_Z=dim_Z, arch=args.arch)
    elif args.model == 'glvae':
        model = GLVAE(channels=nchannels[args.dataset], dim_z=dim_z, dim_Z=dim_Z, arch=args.arch)

    state_dict = torch.load('./results/' + name + '/checkpoints/checkpoint_' + str(epoch) + '.pth',
                            map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    local_traversal = Traversal(zmax=zmax, steps=steps).build(model, batch, 'epoch_' + str(epoch) + '/')
    global_traversal = GlobalTraversal(Zmax=zmax, steps=steps).build(model, batch, 'epoch_' + str(epoch) + '/')
