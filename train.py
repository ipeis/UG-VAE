from models import *
import argparse
from torch import optim
from datasets import *
from torchvision.utils import save_image
from torch.distributions import MultivariateNormal as Normal
import os
import matplotlib.pyplot as plt


#----------------------------------------------------------------------------------------------------------------------#
# Arguments
parser = argparse.ArgumentParser(description='Train UG-VAE')
parser.add_argument('--dim_z', type=int, default=20, metavar='N',
                    help='Dimensions for local latent z (default: 20)')
parser.add_argument('--dim_beta', type=int, default=20, metavar='N',
                    help='Dimensions for global latent beta (default: 20)')
parser.add_argument('--K', type=int, default=20, metavar='N',
                    help='Number of components for the Gaussian mixture (default: 20)')
parser.add_argument('--var_x', type=float, default=2e-1, metavar='N',
                    help='Variance of p(x|z,beta) (default: 2e-1)')
parser.add_argument('--dataset', type=str, default='celeba',
                    help='Name of the dataset (default: celeba)')
parser.add_argument('--arch', type=str, default='beta_vae',
                    help='Architecture for the model (default: beta_vae)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of training epochs (default: 50)')
parser.add_argument('--save_each', type=int, default=1, metavar='N',
                    help='save model and figures each _ epochs (default: 1)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training (default: False)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status (default: 10)')
parser.add_argument('--model_name', type=str, default='celeba',
                    help='name for the model to be saved (default: celeba)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


#----------------------------------------------------------------------------------------------------------------------#
# Creating log dirs
if os.path.isdir('results/' + args.model_name + '/checkpoints/') == False:
    os.makedirs('results/' + args.model_name + '/checkpoints/')
if os.path.isdir('results/' + args.model_name + '/figs/') == False:
    os.makedirs('results/' + args.model_name + '/figs/')
if os.path.isdir('results/' + args.model_name + '/figs/reconstructions') == False:
    os.makedirs('results/' + args.model_name + '/figs/reconstructions')
if os.path.isdir('results/' + args.model_name + '/figs/samples/') == False:
    os.makedirs('results/' + args.model_name + '/figs/samples/')


#----------------------------------------------------------------------------------------------------------------------#
# Loading data
data_tr, _,  data_test = get_data(args.dataset)
train_loader = torch.utils.data.DataLoader(data_tr, batch_size = args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(data_test, batch_size = args.batch_size, shuffle=True)


#----------------------------------------------------------------------------------------------------------------------#
# Model parameters
dim_z = args.dim_z
dim_beta = args.dim_beta
K = args.K
var_x = args.var_x
nchannels = nchannels[args.dataset]

#----------------------------------------------------------------------------------------------------------------------#
# Create model
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
model = UGVAE(
    channels=nchannels, dim_z=dim_z, dim_beta=dim_beta, K=K, var_x=var_x, arch=args.arch, device=device).to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-3)


#----------------------------------------------------------------------------------------------------------------------#
# Train epoch
def train_epoch(model, epoch, train_loader, optimizer, cuda=args.cuda, log_interval=args.log_interval):
    """
    Train a given UGVAE model for one epoch
    Args:
        model: UGVAE object
        epoch: index of the epoch
        train_loader: loader object to extract train batches
        optimizer: optim object to optimize the model
        cuda: flag for using cuda
        log_interval: print log each n.batches

    Returns: train_loss (-ELBO), train_rec p(x|z,beta), KL(q(z|x)|p(z|beta,d)),
                KL(q(d|x)|p(d)), KL(q(beta|x)|p(beta))

    """
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model.train()
    train_loss = 0
    train_rec = 0
    train_klz = 0
    train_kld = 0
    train_klbeta = 0
    nims = len(train_loader.dataset)
    if args.dataset=='celeba_faces' or args.dataset=='cars_chairs':
        #Reset loader each epoch
        data_tr.reset()

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device).view(-1, nchannels, data.shape[-2], data.shape[-1])
        optimizer.zero_grad()
        mu_x, mu_z, var_z, mus_z, vars_z, mu_beta, var_beta, pi = model(data)
        loss, rec, klz, kld, klbeta = model.loss_function(
            data, mu_x, mu_z, var_z, mus_z, vars_z, mu_beta, var_beta, pi)
        loss.backward()
        train_loss += loss.item()
        train_rec += rec.item()
        train_klz += klz.item()
        train_kld += kld.item()
        train_klbeta += klbeta.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), nims,
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))

    train_loss /= nims
    train_rec /= nims
    train_klz /= nims
    train_kld /= nims
    train_klbeta /= nims

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss))

    return train_loss, train_rec, train_klz, train_kld, train_klbeta

#----------------------------------------------------------------------------------------------------------------------#
# Test
def test(model, epoch, test_loader, cuda=args.cuda, model_name='model'):
    """
    Test a given UGVAE model
    Args:
        model: UGVAE object
        epoch: index of the epoch
        test_loader: loader object to extract test batches
        optimizer: optim object to optimize the model
        cuda: flag for using cuda

    Returns: test_loss (-ELBO), test_rec p(x|z,beta), KL(q(z|x)|p(z|beta,d)),
                KL(q(d|x)|p(d)), KL(q(beta|x)|p(beta))

    """
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model.eval()
    test_loss = 0
    test_rec = 0
    test_klz = 0
    test_kld = 0
    test_klbeta = 0
    nims = len(test_loader.dataset)
    with torch.no_grad():
        if args.dataset=='celeba_faces' or args.dataset=='cars_chairs':
            # Reset loader each epoch
            data_test.reset()

        for i, (data, _) in enumerate(test_loader):
            data = data.to(device).view(-1, nchannels, data.shape[-2], data.shape[-1])
            mu_x, mu_z, var_z, mus_z, vars_z, mu_beta, var_beta, pi = model(data)
            loss, rec, klz, kld, klbeta = model.loss_function(
                data, mu_x, mu_z, var_z, mus_z, vars_z, mu_beta, var_beta, pi)
            test_loss += loss.item()
            test_rec += rec.item()
            test_klz += klz.item()
            test_kld += kld.item()
            test_klbeta += klbeta.item()

            if i == 0 and (np.mod(epoch, args.save_each)==0 or epoch==1 or epoch==args.epochs):
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        mu_x[:n]])
                save_image(comparison.cpu(),
                           'results/' + model_name + '/figs/reconstructions/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= nims
    test_rec /= nims
    test_klz /= nims
    test_kld /= nims
    test_klbeta /= nims
    print('====> Test set loss: {:.4f}'.format(test_loss))

    return test_loss, test_rec, test_klz, test_kld, test_klbeta


def plot_losses(tr_losses, test_losses, tr_recs, test_recs,
                tr_klzs, test_klzs,
                tr_klds, test_klds,
                tr_klbetas, test_klbetas,
                model_name='model'):
    """
    Plot training and test losses
    Args:
        tr_losses:      list with tr_losses
        test_losses:    list with test_losses
        tr_recs:        list with tr_recs
        test_recs:      list with test_recs
        tr_klzs:        list with tr_klzs
        test_klzs:      list with test_klzs
        tr_klds:        list with tr_klds
        test_klds:      list with test_klds
        tr_klbetas:     list with tr_klbetas
        test_klbetas:   list with test_klbetas
        model_name:     model_name for saving figures

    Returns:

    """
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    plt.figure()
    if np.mean(tr_losses[:10])>=0:
        plt.semilogy(tr_losses, color=colors[0], label='train_loss')
        plt.semilogy(test_losses, color=colors[0], linestyle=':')
        plt.semilogy(tr_recs, color=colors[1], label='train_rec')
        plt.semilogy(test_recs, color=colors[1], linestyle=':')
        plt.semilogy(tr_klzs, color=colors[2], label=r'$KL_z$')
        plt.semilogy(test_klzs, color=colors[2], linestyle=':')
        plt.semilogy(tr_klds, color=colors[4], label=r'$KL_d$')
        plt.semilogy(test_klds, color=colors[4], linestyle=':')
        plt.semilogy(tr_klbetas, color=colors[5], label=r'$KL_\beta$')
        plt.semilogy(test_klbetas, color=colors[5], linestyle=':')
    else:
        plt.plot(tr_losses, color=colors[0], label='train_loss')
        plt.plot(test_losses, color=colors[0], linestyle=':')
        plt.plot(tr_recs, color=colors[1], label='train_rec')
        plt.plot(test_recs, color=colors[1], linestyle=':')
        plt.plot(tr_klzs, color=colors[2], label=r'$KL_z$')
        plt.plot(test_klzs, color=colors[2], linestyle=':')
        plt.plot(tr_klds, color=colors[4], label=r'$KL_d$')
        plt.plot(test_klds, color=colors[4], linestyle=':')
        plt.plot(tr_klbetas, color=colors[5], label=r'$KL_\beta$')
        plt.plot(test_klbetas, color=colors[5], linestyle=':')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('mean loss')
    plt.legend(loc='best')
    plt.savefig('results/' + model_name + '/figs/losses.pdf')


def save_model(model, epoch, model_name='model'):
    """
    Sabing model to path
    Args:
        model: UGVAE object to save
        epoch: index of epoch
        model_name: model will be saved in results/model_name

    Returns:

    """
    folder = 'results/' + model_name + '/checkpoints/'
    if os.path.isdir(folder) == False:
        os.makedirs(folder)

    torch.save(model.state_dict(), folder + '/checkpoint_' + str(epoch) + '.pth')


#----------------------------------------------------------------------------------------------------------------------#
# Main: train/test for arg.epochs
if __name__ == "__main__":

    tr_losses = []
    tr_recs = []
    tr_klzs = []
    tr_klds = []
    tr_klbetas = []
    test_losses = []
    test_recs = []
    test_klzs = []
    test_klds = []
    test_klbetas = []

    for epoch in range(1, args.epochs + 1):

        # Train
        tr_loss, tr_rec, tr_klz, tr_kld, tr_klbeta = train_epoch(
            model, epoch, train_loader, optimizer,cuda=args.cuda, log_interval=args.log_interval)
        tr_losses.append(tr_loss)
        tr_recs.append(tr_rec)
        tr_klzs.append(tr_klz)
        tr_klds.append(tr_kld)
        tr_klbetas.append(tr_klbeta)

        # Test
        test_loss, test_rec, test_klz, test_kld, test_klbeta = test(
            model, epoch, test_loader, cuda=args.cuda, model_name=args.model_name)
        test_losses.append(test_loss)
        test_recs.append(test_rec)
        test_klzs.append(test_klz)
        test_klds.append(test_kld)
        test_klbetas.append(test_klbeta)

        with torch.no_grad():
            losses = {
                'tr_losses': tr_losses,
                'test_losses': test_losses,
                'tr_recs': tr_recs,
                'test_recs': test_recs,
                'tr_klzs': tr_klzs,
                'test_klzs': test_klzs,
                'tr_klds': tr_klds,
                'test_klds': test_klds,
                'tr_klbetas': tr_klbetas,
                'test_klbetas': test_klbetas
            }
            # Save losses and figures
            np.save('results/' + args.model_name + '/checkpoints/losses', losses)
            plot_losses(tr_losses, test_losses, tr_recs, test_recs,
                        tr_klzs, test_klzs, tr_klds, test_klds,
                        tr_klbetas, test_klbetas,
                        model_name=args.model_name)

            # Log
            if np.mod(epoch, args.save_each)==0 or epoch==1 or epoch==args.epochs:

                # Save samples with fixed d
                sample_beta = torch.randn(dim_beta).to(device)
                mus_z, vars_z = model._z_prior(sample_beta)
                samples_z = torch.stack([torch.stack(
                    [Normal(mu_z, torch.diag(var_z)).sample().to(device) for mu_z, var_z in zip(mus_z, vars_z)]) for i
                    in range(64)])  # [64, K, dim_z] # each batch in this list has d fixed
                samples = [model._decode(samples_z[:, k], sample_beta ) for k in range(K)]
                # Save each batch as grid image
                [save_image(samples[k],
                            'results/' + args.model_name + '/figs/samples/sample_' + str(epoch) + '_L' + str(k) + '.png') for
                 k in range(K)]
                plt.close('all')
                # Save the model
                save_model(model, epoch, model_name=args.model_name)