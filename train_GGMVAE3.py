from models import *
import argparse
from torchvision import datasets, transforms
from torch import nn, optim
from datasets import *
from torchvision.utils import save_image
from torch.distributions import MultivariateNormal as Normal

########################################################################################################################
parser = argparse.ArgumentParser(description='Train GGMVAE')
parser.add_argument('--dim_z', type=int, default=10, metavar='N',
                    help='Dimensions for local latent')
parser.add_argument('--dim_beta', type=int, default=20, metavar='N',
                    help='Dimensions for global latent')
parser.add_argument('--dim_w', type=int, default=20, metavar='N',
                    help='Dimensions for global latent')
parser.add_argument('--K', type=int, default=10, metavar='N',
                    help='Number of components for the Gaussian Global mixture')
parser.add_argument('--dataset', type=str, default='celeba_lfw',
                    help='Name of the dataset')
parser.add_argument('--arch', type=str, default='beta_vae',
                    help='Architecture for the model')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--save_each', type=int, default=1, metavar='N',
                    help='save model and figures each _ epochs (default: 10)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model_name', type=str, default='GGMVAE3/celeba_lfw',
                    help='name for the model to be saved')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
########################################################################################################################

torch.manual_seed(args.seed)

model_name = args.model_name

device = torch.device("cuda" if args.cuda else "cpu")

if os.path.isdir('results/' + model_name + '/checkpoints/') == False:
    os.makedirs('results/' + model_name + '/checkpoints/')

if os.path.isdir('results/' + model_name + '/figs/') == False:
    os.makedirs('results/' + model_name + '/figs/')

if os.path.isdir('results/' + model_name + '/figs/reconstructions') == False:
    os.makedirs('results/' + model_name + '/figs/reconstructions')

if os.path.isdir('results/' + model_name + '/figs/samples/') == False:
    os.makedirs('results/' + model_name + '/figs/samples/')


#kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

########################################################################################################################
if args.dataset=='mnist_series':
    assert (args.dataset=='mnist_series' and args.batch_size == 128), 'mnist_series dataset is only for batches of 128 images.'

data_tr, _,  data_test = get_data(args.dataset)

"""
if dataset=='celebA':

    data_tr = CelebA('./data/', split="train",
                     transform=transforms.ToTensor(), download=False)   #download=True for downloading the dataset
    data_test = CelebA('./data/', split="test",
                       transform=transforms.ToTensor(), download=False)
if dataset=='mnist':
    data_tr = datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor())
    data_test = datasets.MNIST('../data', train=False, download=True,
                   transform=transforms.ToTensor())
"""

if args.dataset == 'mnist_series' or args.dataset == 'mnist_svhn_series':
    batch_size = 1
else:
    batch_size = args.batch_size

train_loader = torch.utils.data.DataLoader(data_tr, batch_size = batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(data_test, batch_size = batch_size, shuffle=True)
########################################################################################################################


dim_z = args.dim_z
dim_beta = args.dim_beta
dim_w = args.dim_w
K = args.K

distribution = distributions[args.dataset]
nchannels = nchannels[args.dataset]


model = GGMVAE3(channels=nchannels, dim_z=dim_z, dim_beta=dim_beta, dim_w=dim_w, K=K, arch=args.arch, device=device).to(device)

optimizer = optim.Adam(model.parameters(), lr=5e-3)


########################################################################################################################
########################################################################################################################
########################################################################################################################

def train_epoch(model, epoch, train_loader, optimizer, cuda=False, log_interval=10):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model.train()
    train_loss = 0
    train_rec = 0
    train_klz = 0
    train_klalpha = 0
    train_klw = 0
    nims = len(train_loader.dataset)
    if args.dataset=='mnist_svhn' or args.dataset=='celeba_faces' or args.dataset=='celeba_faces_batch' or args.dataset=='celeba_lfw':
        #Reset loader each epoch
        data_tr.reset()
        #train_loader = torch.utils.data.DataLoader(data_tr, batch_size=args.batch_size, shuffle=True)
    elif args.dataset=='mnist_series' or args.dataset=='mnist_svhn_series':
        data_tr.reset()
        nims = data_tr.nbatches * data_tr.batch_size


    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device).view(-1, nchannels, data.shape[-2], data.shape[-1])
        optimizer.zero_grad()
        recon_batch, mu_z, var_z, mu_beta, var_beta, mus_z_p, vars_z_p, mu_w, var_w, pi = model(data)
        loss, rec, klz, klalpha, klw = model.loss_function(recon_batch, data, mu_z, var_z, mus_z_p, vars_z_p, mu_w, var_w, pi, distribution)
        loss.backward()
        train_loss += loss.item()
        train_rec += rec.item()
        train_klz += klz.item()
        train_klalpha += klalpha.item()
        train_klw += klw.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), nims,
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / nims))

    return train_loss / nims, train_rec / nims, \
           train_klz / nims, train_klalpha / nims, train_klw / nims


def test(model, epoch, test_loader, cuda=False, model_name='model'):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model.eval()
    test_loss = 0
    test_rec = 0
    test_klz = 0
    test_klalpha = 0
    test_klw = 0
    nims = len(test_loader.dataset)
    with torch.no_grad():
        if args.dataset == 'mnist_svhn' or args.dataset=='celeba_faces' or args.dataset=='celeba_faces_batch' or args.dataset=='celeba_lfw':
            # Reset loader each epoch
            data_test.reset()
            #test_loader = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=True)
        elif args.dataset=='mnist_series' or args.dataset=='mnist_svhn_series':
            data_test.reset()
            nims = data_test.nbatches * data_test.batch_size
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device).view(-1, nchannels, data.shape[-2], data.shape[-1])
            recon_batch, mu_z, var_z, mu_beta, var_beta, mus_z_p, vars_z_p, mu_w, var_w, pi = model(data)
            loss, rec, klz, klalpha, klw = model.loss_function(recon_batch, data, mu_z, var_z, mus_z_p, vars_z_p,
                                                                       mu_w, var_w, pi, distribution)
            test_loss += loss.item()
            test_rec += rec.item()
            test_klz += klz.item()
            test_klalpha += klalpha.item()
            test_klw += klw.item()

            if i == 0 and (np.mod(epoch, args.save_each)==0 or epoch==1 or epoch==args.epochs):
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n]] +
                                        [recon_batch[k][:n] for k in range(args.K)])
                save_image(comparison.cpu(),
                           'results/' + model_name + '/figs/reconstructions/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= nims
    test_rec /= nims
    test_klz /= nims
    test_klalpha /= nims
    test_klw /= nims
    print('====> Test set loss: {:.4f}'.format(test_loss))

    return test_loss, test_rec, test_klz, test_klalpha, test_klw




def plot_losses(model, tr_losses, test_losses, tr_recs, test_recs,
                tr_klzs, test_klzs,
                tr_klalphas, test_klalphas,
                tr_klws, test_klws,
                model_name='model'):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    plt.figure()
    plt.semilogy(tr_losses, color=colors[0], label='train_loss')
    plt.semilogy(test_losses, color=colors[0], linestyle=':')
    plt.semilogy(tr_recs, color=colors[1], label='train_rec')
    plt.semilogy(test_recs, color=colors[1], linestyle=':')
    plt.semilogy(tr_klzs, color=colors[2], label=r'$KL_z$')
    plt.semilogy(test_klzs, color=colors[2], linestyle=':')
    plt.semilogy(tr_klalphas, color=colors[4], label=r'$KL_\alpha$')
    plt.semilogy(test_klalphas, color=colors[4], linestyle=':')
    plt.semilogy(tr_klws, color=colors[5], label=r'$KL_w$')
    plt.semilogy(test_klws, color=colors[5], linestyle=':')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('mean loss')
    plt.legend(loc='best')
    plt.savefig('results/' + model_name + '/figs/losses.pdf')


def save_model(model, epoch, model_name='model'):
    folder = 'results/' + model_name + '/checkpoints/'
    if os.path.isdir(folder) == False:
        os.makedirs(folder)

    torch.save(model.state_dict(), folder + '/checkpoint_' + str(epoch) + '.pth')


########################################################################################
########################################################################################
########################################################################################


if __name__ == "__main__":

    tr_losses = []
    tr_recs = []
    tr_klzs = []
    tr_klalphas = []
    tr_klws = []
    test_losses = []
    test_recs = []
    test_klzs = []
    test_klalphas = []
    test_klws = []

    for epoch in range(1, args.epochs + 1):

        tr_loss, tr_rec, tr_klz, tr_klalpha, tr_klw = train_epoch(model, epoch, train_loader, optimizer,
                                                              cuda=args.cuda, log_interval=args.log_interval)
        tr_losses.append(tr_loss)
        tr_recs.append(tr_rec)
        tr_klzs.append(tr_klz)
        tr_klalphas.append(tr_klalpha)
        tr_klws.append(tr_klw)

        test_loss, test_rec, test_klz, test_klalpha, test_klw = test(model, epoch, test_loader,
                                                               cuda=args.cuda, model_name=model_name)
        test_losses.append(test_loss)
        test_recs.append(test_rec)
        test_klzs.append(test_klz)
        test_klalphas.append(test_klalpha)
        test_klws.append(test_klw)

        # Save figures
        with torch.no_grad():

            losses = {
                'tr_losses': tr_losses,
                'test_losses': test_losses,
                'tr_recs': tr_recs,
                'test_recs': test_recs,
                'tr_klzs': tr_klzs,
                'test_klzs': test_klzs,
                'tr_klalphas': tr_klalphas,
                'test_klalphas': test_klalphas,
                'tr_klws': tr_klws,
                'test_klws': test_klws
            }
            np.save('results/' + model_name + '/checkpoints/losses', losses)
            plot_losses(model, tr_losses, test_losses, tr_recs, test_recs,
                        tr_klzs, test_klzs, tr_klalphas, test_klalphas,
                        tr_klws, test_klws,
                        model_name=model_name)
            if np.mod(epoch, args.save_each)==0 or epoch==1 or epoch==args.epochs:

                sample_z = torch.randn(64, dim_z).to(device)
                sample_w = torch.randn(1, dim_w).to(device)
                for k in range(K):
                    #k = np.random.choice(a=np.arange(K), p=model.pi_alpha.detach().squeeze().numpy())
                    out = model.beta_gen[k](sample_w)
                    mu_beta = out[:, :dim_beta]
                    var_beta = torch.exp(torch.tanh(out[:, dim_beta:]))
                    sample_beta = Normal(mu_beta, torch.diag(var_beta.squeeze())).sample().view(dim_beta).to(device)
                    sample = model._decode(sample_z, sample_beta).to(device)
                    save_image(sample,
                           'results/' + model_name + '/figs/samples/sample_' + str(epoch) + '_K' + str(k) + '.png')
                plt.close('all')

                save_model(model, epoch, model_name=model_name)
                #plot_latent(model, epoch, test_loader, cuda=args.cuda, model_name=model_name)
                #plot_global_latent(model, epoch, nsamples=4, nreps=1000, nims=20, cuda=args.cuda, model_name=model_name)


