from GLVAE import *
import argparse
from torchvision import datasets, transforms
from torch import nn, optim


########################################################################################
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--dim_z', type=int, default=10, metavar='N',
                    help='Dimensions for local latent')
parser.add_argument('--dim_Z', type=int, default=10, metavar='N',
                    help='Dimensions for global latent')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model_name', type=str, default='trained',
                    help='name for the model to be saved')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
########################################################################################

torch.manual_seed(args.seed)

model_name = args.model_name
model_name = 'prueba'

device = torch.device("cuda" if args.cuda else "cpu")

if os.path.isdir('results/' + model_name + '/checkpoints/') == False:
    os.makedirs('results/' + model_name + '/checkpoints/')

    if os.path.isdir('results/' + model_name + '/figs/') == False:
        os.makedirs('results/' + model_name + '/figs/')

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

########################################################################################
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
########################################################################################


dim_z = args.dim_z
dim_Z = args.dim_Z
dim_z = 2
dim_Z = 20

model = GLVAE(dim_z, dim_Z).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)



if __name__ == "__main__":

    tr_losses = []
    tr_recs = []
    tr_kl_ls = []
    tr_kl_gs = []
    test_losses = []
    test_recs = []
    test_kl_ls = []
    test_kl_gs = []

    for epoch in range(1, args.epochs + 1):

        tr_loss, tr_rec, tr_kl_l, tr_kl_g = model.train_epoch(epoch, train_loader, optimizer,
                                                              cuda=args.cuda, log_interval=args.log_interval)
        tr_losses.append(tr_loss)
        tr_recs.append(tr_rec)
        tr_kl_ls.append(tr_kl_l)
        tr_kl_gs.append(tr_kl_g)

        test_loss, test_rec, test_kl_l, test_kl_g = model.test(epoch, test_loader,
                                                               cuda=args.cuda, model_name=model_name)
        test_losses.append(test_loss)
        test_recs.append(test_rec)
        test_kl_ls.append(test_kl_l)
        test_kl_gs.append(test_kl_g)

        # Save figures
        with torch.no_grad():
            sample_z = torch.randn(64, dim_z).to(device)
            sample_Z = torch.randn(dim_Z).to(device)
            sample = model.decode(sample_z, sample_Z).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/' + model_name + '/figs/sample_' + str(epoch) + '.png')
            plt.close('all')
            model.plot_losses(tr_losses, test_losses, tr_recs, test_recs, tr_kl_ls, test_kl_ls, tr_kl_gs, test_kl_gs,
                    model_name=model_name)
            if np.mod(epoch, 100)==0 or epoch==1 or epoch==args.epochs:
                model.save_model(epoch, model_name=model_name)
                model.plot_latent(epoch, test_loader, cuda=args.cuda, model_name=model_name)
                model.plot_global_latent(epoch, nsamples=4, nreps=1000, nims=20, cuda=args.cuda, model_name=model_name)


