from models import *
import argparse
from torch import optim
from datasets import *
from torchvision.utils import save_image

########################################################################################################################
parser = argparse.ArgumentParser(description='Train betaVAE')
parser.add_argument('--dim_z', type=int, default=20, metavar='N',
                    help='dimensions for local latent')
parser.add_argument('--dim_w', type=int, default=20, metavar='N',
                    help='dimensions for local noise latent')
parser.add_argument('--K', type=int, default=2, metavar='N',
                    help='Number of components for the Gaussian mixture')
parser.add_argument('--var_x', type=float, default=2e-1, metavar='N',
                    help='Variance of the data')
parser.add_argument('--dataset', type=str, default='celeba',
                    help='name of the dataset')
parser.add_argument('--arch', type=str, default='beta_vae',
                    help='architecture of the model')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of training epochs (default: 500)')
parser.add_argument('--beta', type=float, default=10.0, metavar='N',
                    help='value for disentanglement factor')
parser.add_argument('--save_each', type=int, default=1, metavar='N',
                    help='save model and figures each _ epochs (default: 10)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model_name', type=str, default='GMVAE/celeba',
                    help='name for results folder')
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



########################################################################################################################
data_tr, _,  data_test = get_data(args.dataset)
train_loader = torch.utils.data.DataLoader(data_tr, batch_size = args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(data_test, batch_size = args.batch_size, shuffle=True)
########################################################################################################################


dim_z = args.dim_z
dim_w = args.dim_w
K = args.K

nchannels = nchannels[args.dataset]

model = GMVAE(channels=nchannels, dim_z=dim_z, dim_w=dim_w, K=K, arch=args.arch).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

beta = torch.tensor(args.beta).to(device)

########################################################################################################################
########################################################################################################################
########################################################################################################################

def train_epoch(model, epoch, train_loader, optimizer, cuda=False, log_interval=10):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model.train()
    train_loss = 0
    train_rec = 0
    train_kl_z = 0
    train_kl_w = 0
    train_kl_d = 0
    nims = len(train_loader.dataset)

    if args.dataset=='celeba_faces' or args.dataset=='fashionmnist_shoes':
        #Reset loader each epoch
        data_tr.reset()

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device).view(-1, nchannels, data.shape[-2], data.shape[-1])
        optimizer.zero_grad()
        recon_batch, mu_z, var_z, mu_w, var_w, pi, muz_z, vars_z = model.forward(data)
        loss, rec, kl_z, kl_w, kl_d = model.loss_function(recon_batch, data, mu_z, var_z, mu_w, var_w, pi, muz_z, vars_z)
        loss.backward()
        train_loss += loss.item()
        train_rec += rec.item()
        train_kl_z += kl_z.item()
        train_kl_w += kl_w.item()
        train_kl_d += kl_d.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), nims,
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / nims))

    return train_loss / nims, train_rec / nims, \
           train_kl_z / nims, train_kl_w / nims, train_kl_d / nims


def test(model, epoch, test_loader, cuda=False, model_name='model'):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model.eval()
    test_loss = 0
    test_rec = 0
    test_kl_z = 0
    test_kl_w = 0
    test_kl_d = 0
    nims = len(test_loader.dataset)

    if args.dataset=='celeba_faces' or args.dataset=='fashionmnist_shoes':
        #Reset loader each epoch
        data_test.reset()

    with torch.no_grad():

        for i, (data, _) in enumerate(test_loader):
            data = data.to(device).view(-1, nchannels, data.shape[-2], data.shape[-1])
            recon_batch, mu_z, var_z, mu_w, var_w, pi, muz_z, vars_z = model.forward(data)
            loss, rec, kl_z, kl_w, kl_d = model.loss_function(recon_batch, data, mu_z, var_z, mu_z, var_w, pi, muz_z, vars_z)
            test_loss += loss.item()
            test_rec += rec.item()
            test_kl_z += kl_z.item()
            test_kl_w += kl_w.item()
            test_kl_d += kl_d.item()

            if i == 0 and (np.mod(epoch, args.save_each)==0 or epoch==1 or epoch==args.epochs):
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch[:n]])
                save_image(comparison.cpu(),
                           'results/' + model_name + '/figs/reconstructions/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= nims
    test_rec /= nims
    test_kl_z /= nims
    test_kl_w /= nims
    test_kl_d /= nims
    print('====> Test set loss: {:.4f}'.format(test_loss))

    return test_loss, test_rec, test_kl_z, test_kl_w, test_kl_d




def plot_losses(tr_losses, test_losses, tr_recs, test_recs, tr_klzs, test_klzs, tr_klws, test_klws, tr_klds, test_klds,
                model_name='model'):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    plt.figure()
    plt.semilogy(tr_losses, color=colors[0], label='train_loss')
    plt.semilogy(test_losses, color=colors[0], linestyle=':')
    plt.semilogy(tr_recs, color=colors[1], label='train_rec')
    plt.semilogy(test_recs, color=colors[1], linestyle=':')
    plt.semilogy(tr_klzs, color=colors[2], label='KL_z')
    plt.semilogy(test_klzs, color=colors[2], linestyle=':')
    plt.semilogy(tr_klws, color=colors[2], label='KL_w')
    plt.semilogy(test_klws, color=colors[2], linestyle=':')
    plt.semilogy(tr_klds, color=colors[2], label='KL_d')
    plt.semilogy(test_klds, color=colors[2], linestyle=':')
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
    tr_kl_zs = []
    tr_kl_ws = []
    tr_kl_ds = []
    test_losses = []
    test_recs = []
    test_kl_zs = []
    test_kl_ws = []
    test_kl_ds = []

    for epoch in range(1, args.epochs + 1):

        tr_loss, tr_rec, tr_kl_z, tr_kl_w, tr_kl_d = train_epoch(model, epoch, train_loader, optimizer,
                                                              cuda=args.cuda, log_interval=args.log_interval)
        tr_losses.append(tr_loss)
        tr_recs.append(tr_rec)
        tr_kl_zs.append(tr_kl_z)
        tr_kl_ws.append(tr_kl_w)
        tr_kl_ds.append(tr_kl_d)

        test_loss, test_rec, test_kl_z, test_kl_w, test_kl_d = test(model, epoch, test_loader,
                                                               cuda=args.cuda, model_name=model_name)
        test_losses.append(test_loss)
        test_recs.append(test_rec)
        test_kl_zs.append(test_kl_z)
        test_kl_ws.append(test_kl_w)
        test_kl_ds.append(test_kl_d)

        # Save figures
        with torch.no_grad():
            losses = {
                'tr_losses': tr_losses,
                'test_losses': test_losses,
                'tr_recs': tr_recs,
                'test_recs': test_recs,
                'tr_kl_zs': tr_kl_zs,
                'test_kl_zs': test_kl_zs,
                'tr_kl_ws': tr_kl_ws,
                'test_kl_ws': test_kl_ws,
                'tr_kl_ds': tr_kl_ds,
                'test_kl_ds': test_kl_ds,
            }
            np.save('results/' + model_name + '/checkpoints/losses', losses)
            plot_losses(tr_losses, test_losses, tr_recs, test_recs, tr_kl_zs, test_kl_zs, tr_kl_ws, test_kl_ws, tr_kl_ds, test_kl_ds,
                    model_name=model_name)
            if np.mod(epoch, args.save_each)==0 or epoch==1 or epoch==args.epochs:
                sample_w = torch.randn(64, dim_z).to(device)

                mus_z, vars_z = model._z_prior(sample_w)
                for k in range(K):
                    sample_z = model.reparameterize(mus_z[k], vars_z[k]).cpu():
                    sample = model._decode(sample_z).cpu()
                    save_image(sample,
                               'results/' + model_name + '/figs/samples/sample_' + str(epoch) + '.png')
                    plt.close('all')
                    save_model(model, epoch, model_name=model_name)



