from models import *
import argparse
from torchvision import datasets, transforms
from torch import nn, optim
from datasets import *
from torchvision.utils import save_image
from torch.distributions import MultivariateNormal as Normal

########################################################################################################################
parser = argparse.ArgumentParser(description='Train DGMVAE')
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
parser.add_argument('--dataset', type=str, default='mnist_series',
                    help='Name of the dataset')
parser.add_argument('--arch', type=str, default='k_vae',
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
parser.add_argument('--model_name', type=str, default='DGMVAE/mnist_series2',
                    help='name for the model to be saved')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
########################################################################################################################

torch.manual_seed(args.seed)
model_name = args.model_name
device = torch.device("cuda" if args.cuda else "cpu")

if os.path.isdir('results/' + model_name ) == False:
    os.makedirs('results/' + model_name + '/checkpoints/')
    os.makedirs('results/' + model_name + '/figs/')
    os.makedirs('results/' + model_name + '/figs/reconstructions')
    os.makedirs('results/' + model_name + '/figs/samples/')


########################################################################################################################
if args.dataset=='mnist_series':
    assert (args.dataset=='mnist_series' and args.batch_size == 128), 'mnist_series dataset is only for batches of 128 images.'

data_tr, _,  data_test = get_data(args.dataset)

if args.dataset == 'mnist_series' or args.dataset == 'mnist_svhn_series':
    batch_size = 1
else:
    batch_size = args.batch_size

train_loader = torch.utils.data.DataLoader(data_tr, batch_size = batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(data_test, batch_size = batch_size, shuffle=True)
########################################################################################################################


dim_z = args.dim_z
L = args.L
dim_beta = args.dim_beta
K = args.K
dim_w = args.dim_w
var_x = args.var_x

distribution = distributions[args.dataset]
nchannels = nchannels[args.dataset]


model = DGMVAE(channels=nchannels, dim_z=dim_z, L=L, dim_beta=dim_beta, K=K, dim_w=dim_w, var_x=var_x, arch=args.arch, device=device).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)


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
    train_kld = 0
    train_klw = 0
    train_klalpha = 0
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
        mus_x, mu_z, var_z, mus_z, vars_z, pi_d, mu_w, var_w, mus_beta, vars_beta, pi_alpha = model(data)
        loss, rec, klz, kld, klw, klalpha = model.loss_function(data, mus_x, mu_z, var_z, mus_z, vars_z, pi_d, mu_w, var_w, pi_alpha)
        loss.backward()
        train_loss += loss.item()
        train_rec += rec.item()
        train_klz += klz.item()
        train_kld += kld.item()
        train_klw += klw.item()
        train_klalpha += klalpha.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), nims,
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / nims))

    train_loss /= nims
    train_rec /= nims
    train_klz /= nims
    train_kld /= nims
    train_klw /= nims
    train_klalpha /= nims

    return train_loss , train_rec, train_klz, train_kld, train_klw, train_klalpha


def test(model, epoch, test_loader, cuda=False, model_name='model'):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model.eval()
    test_loss = 0
    test_rec = 0
    test_klz = 0
    test_kld = 0
    test_klw = 0
    test_klalpha = 0
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
            mus_x, mu_z, var_z, mus_z, vars_z, pi_d, mu_w, var_w, mus_beta, vars_beta, pi_alpha = model(data)
            loss, rec, klz, kld, klw, klalpha = model.loss_function(data, mus_x, mu_z, var_z, mus_z, vars_z, pi_d, mu_w, var_w,
                                                              pi_alpha)
            test_loss += loss.item()
            test_rec += rec.item()
            test_klz += klz.item()
            test_kld += kld.item()
            test_klw += klw.item()
            test_klalpha += klalpha.item()

            if i == 0 and (np.mod(epoch, args.save_each)==0 or epoch==1 or epoch==args.epochs):
                n = min(data.size(0), 8)
                comparison = [torch.cat([data[:n], mu_x[:n]]) for mu_x in mus_x]
                [save_image(comparison[k].cpu(),
                            'results/' + model_name + '/figs/reconstructions/reconstruction_' + str(epoch) + '_K' + str(
                                k) + '.png', nrow=n) for k in range(model.K)]

    test_loss /= nims
    test_rec /= nims
    test_klz /= nims
    test_kld /= nims
    test_klw /= nims
    test_klalpha /= nims
    print('====> Test set loss: {:.4f}'.format(test_loss))

    return test_loss, test_rec, test_klz, test_kld, test_klw, test_klalpha




def plot_losses(tr_losses, test_losses, tr_recs, test_recs,
                tr_klzs, test_klzs,
                tr_klds, test_klds,
                tr_klws, test_klws,
                tr_klalphas, test_klalphas,
                model_name='model'):
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
        plt.semilogy(tr_klds, color=colors[3], label=r'$KL_d$')
        plt.semilogy(test_klds, color=colors[3], linestyle=':')
        plt.semilogy(tr_klws, color=colors[4], label=r'$KL_w$')
        plt.semilogy(test_klws, color=colors[4], linestyle=':')
        plt.semilogy(tr_klalphas, color=colors[5], label=r'$KL_\alpha$')
        plt.semilogy(test_klalphas, color=colors[5], linestyle=':')
    else:
        plt.plot(tr_losses, color=colors[0], label='train_loss')
        plt.plot(test_losses, color=colors[0], linestyle=':')
        plt.plot(tr_recs, color=colors[1], label='train_rec')
        plt.plot(test_recs, color=colors[1], linestyle=':')
        plt.plot(tr_klzs, color=colors[2], label=r'$KL_z$')
        plt.plot(test_klzs, color=colors[2], linestyle=':')
        plt.plot(tr_klds, color=colors[3], label=r'$KL_d$')
        plt.plot(test_klds, color=colors[3], linestyle=':')
        plt.plot(tr_klws, color=colors[4], label=r'$KL_w$')
        plt.plot(test_klws, color=colors[4], linestyle=':')
        plt.plot(tr_klalphas, color=colors[5], label=r'$KL_\alpha$')
        plt.plot(test_klalphas, color=colors[5], linestyle=':')
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
    tr_klds = []
    tr_klws = []
    tr_klalphas = []
    test_losses = []
    test_recs = []
    test_klzs = []
    test_klds = []
    test_klws = []
    test_klalphas = []

    for epoch in range(1, args.epochs + 1):

        tr_loss, tr_rec, tr_klz, tr_kld, tr_klw, tr_klalpha = train_epoch(model, epoch, train_loader, optimizer,
                                                              cuda=args.cuda, log_interval=args.log_interval)
        tr_losses.append(tr_loss)
        tr_recs.append(tr_rec)
        tr_klzs.append(tr_klz)
        tr_klds.append(tr_kld)
        tr_klws.append(tr_klw)
        tr_klalphas.append(tr_klalpha)

        test_loss, test_rec, test_klz, test_kld, test_klw, test_klalpha = test(model, epoch, test_loader,
                                                               cuda=args.cuda, model_name=model_name)
        test_losses.append(test_loss)
        test_recs.append(test_rec)
        test_klzs.append(test_klz)
        test_klds.append(test_kld)
        test_klws.append(test_klw)
        test_klalphas.append(test_klalpha)

        # Save figures
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
                'tr_klws': tr_klws,
                'test_klws': test_klws,
                'tr_klalphas': tr_klalphas,
                'test_klalphas': test_klalphas
            }
            np.save('results/' + model_name + '/checkpoints/losses', losses)
            plot_losses(tr_losses, test_losses, tr_recs, test_recs,
                        tr_klzs, test_klzs, tr_klds, test_klds, tr_klws, test_klws,
                        tr_klalphas, test_klalphas,
                        model_name=model_name)
            if np.mod(epoch, args.save_each)==0 or epoch==1 or epoch==args.epochs:

                sample_w = torch.randn(dim_w).to(device)
                mus_beta, vars_beta = model._beta_mix(sample_w)
                betas = [Normal(mus_beta[k], torch.diag(vars_beta[k])).sample().to(device) for k
                    in range(model.K)]
                theta = [model._z_mix(beta) for beta in betas]
                for l in range(model.L):
                    mus_z = torch.stack([theta[k][0][l, :] for k in range(model.K)])
                    vars_z = torch.stack([theta[k][1][l, :] for k in range(model.K)])
                    samples_z = torch.stack(
                        [ torch.stack( [Normal(mus_z[k], torch.diag(vars_z[k])).sample().to(device) for i
                        in range(10)])
                          for k in range(model.K)]
                    )
                    # [64, K, dim_z]
                    samples = torch.cat([model._decode(samples_z[k], betas[k] ) for k in range(model.K)])
                    save_image(samples,
                                'results/' + model_name + '/figs/samples/sample_' + str(epoch) + '_L' + str(l) + '.png',
                               nrow=10)
                    plt.close('all')

                    save_model(model, epoch, model_name=model_name)
                    #plot_latent(model, epoch, test_loader, cuda=args.cuda, model_name=model_name)
                    #plot_global_latent(model, epoch, nsamples=4, nreps=1000, nims=20, cuda=args.cuda, model_name=model_name)


