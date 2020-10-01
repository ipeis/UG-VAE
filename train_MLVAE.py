from models import *
import argparse
from torchvision import datasets, transforms
from torch import nn, optim
from datasets import *
from torchvision.utils import save_image
from torch.distributions import MultivariateNormal as Normal

########################################################################################################################
parser = argparse.ArgumentParser(description='Train MLVAE')
parser.add_argument('--dim_s', type=int, default=10, metavar='N',
                    help='Dimensions for local latent')
parser.add_argument('--dim_C', type=int, default=20, metavar='N',
                    help='Dimensions for global latent')
parser.add_argument('--dataset', type=str, default='celeba',
                    help='Name of the dataset')
parser.add_argument('--arch', type=str, default='beta_vae',
                    help='Architecture for the model')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--save_each', type=int, default=1, metavar='N',
                    help='save model and figures each _ epochs (default: 10)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model_name', type=str, default='MLVAE/celeba',
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

########################################################################################################################
data_tr, _,  data_test = get_data(args.dataset)
train_loader = torch.utils.data.DataLoader(data_tr, batch_size = args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(data_test, batch_size = args.batch_size, shuffle=True)
########################################################################################################################


nchannels = nchannels[args.dataset]


model = MLVAE(channels=nchannels, dim_s=args.dim_s, dim_C=args.dim_C,arch=args.arch).to(device)

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
    train_kls = 0
    train_klC = 0
    nims = len(train_loader.dataset)
    if args.dataset=='celeba_faces':
        #Reset loader each epoch
        data_tr.reset()

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device).view(-1, nchannels, data.shape[-2], data.shape[-1])
        optimizer.zero_grad()
        mu_x, mu_s, var_s, mu_C, var_C = model(data)
        loss, rec, kls, klC = model.loss_function(data, mu_x, mu_s, var_s, mu_C, var_C)
        loss.backward()
        train_loss += loss.item()
        train_rec += rec.item()
        train_kls += kls.item()
        train_klC += klC.item()
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
    train_kls /= nims
    train_klC /= nims

    return train_loss, train_rec, train_kls, train_klC


def test(model, epoch, test_loader, cuda=False, model_name='model'):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model.eval()
    test_loss = 0
    test_rec = 0
    test_kls = 0
    test_klC = 0
    nims = len(test_loader.dataset)
    with torch.no_grad():
        if args.dataset=='celeba_faces':
            # Reset loader each epoch
            data_test.reset()

        for i, (data, _) in enumerate(test_loader):
            data = data.to(device).view(-1, nchannels, data.shape[-2], data.shape[-1])
            mu_x, mu_s, var_s, mu_C, var_C = model(data)
            loss, rec, kls, klC = model.loss_function(data, mu_x, mu_s, var_s, mu_C, var_C)
            test_loss += loss.item()
            test_rec += rec.item()
            test_kls += kls.item()
            test_klC += klC.item()

            if i == 0 and (np.mod(epoch, args.save_each)==0 or epoch==1 or epoch==args.epochs):
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        mu_x[:n]])
                save_image(comparison.cpu(),
                           'results/' + model_name + '/figs/reconstructions/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= nims
    test_rec /= nims
    test_kls /= nims
    test_klC /= nims
    print('====> Test set loss: {:.4f}'.format(test_loss))

    return test_loss, test_rec, test_kls, test_klC


def plot_losses(tr_losses, test_losses, tr_recs, test_recs,
                tr_klss, test_klss,
                tr_klCs, test_klCs,
                model_name='model'):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    plt.figure()
    if np.mean(tr_losses[:10])>=0:
        plt.semilogy(tr_losses, color=colors[0], label='train_loss')
        plt.semilogy(test_losses, color=colors[0], linestyle=':')
        plt.semilogy(tr_recs, color=colors[1], label='train_rec')
        plt.semilogy(test_recs, color=colors[1], linestyle=':')
        plt.semilogy(tr_klss, color=colors[2], label=r'$KL_s$')
        plt.semilogy(test_klss, color=colors[2], linestyle=':')
        plt.semilogy(tr_klCs, color=colors[3], label=r'$KL_C')
        plt.semilogy(test_klCs, color=colors[3], linestyle=':')
    else:
        plt.plot(tr_losses, color=colors[0], label='train_loss')
        plt.plot(test_losses, color=colors[0], linestyle=':')
        plt.plot(tr_recs, color=colors[1], label='train_rec')
        plt.plot(test_recs, color=colors[1], linestyle=':')
        plt.plot(tr_klss, color=colors[2], label=r'$KL_z$')
        plt.plot(test_klss, color=colors[2], linestyle=':')
        plt.plot(tr_klCs, color=colors[3], label=r'$KL_C')
        plt.plot(test_klCs, color=colors[3], linestyle=':')
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
    tr_klss = []
    tr_klCs = []
    test_losses = []
    test_recs = []
    test_klss = []
    test_klCs = []

    for epoch in range(1, args.epochs + 1):

        tr_loss, tr_rec, tr_kls, tr_klC = train_epoch(model, epoch, train_loader, optimizer,
                                                              cuda=args.cuda, log_interval=args.log_interval)
        tr_losses.append(tr_loss)
        tr_recs.append(tr_rec)
        tr_klss.append(tr_kls)
        tr_klCs.append(tr_klC)

        test_loss, test_rec, test_kls, test_klC = test(model, epoch, test_loader,
                                                               cuda=args.cuda, model_name=model_name)
        test_losses.append(test_loss)
        test_recs.append(test_rec)
        test_klss.append(test_kls)
        test_klCs.append(test_klC)

        # Save figures
        with torch.no_grad():

            losses = {
                'tr_losses': tr_losses,
                'test_losses': test_losses,
                'tr_recs': tr_recs,
                'test_recs': test_recs,
                'tr_klss': tr_klss,
                'test_klss': test_klss,
                'tr_klCs': tr_klCs,
                'test_klCs': test_klCs
            }
            np.save('results/' + model_name + '/checkpoints/losses', losses)
            plot_losses(tr_losses, test_losses, tr_recs, test_recs,
                        tr_klss, test_klss,
                        tr_klCs, test_klCs,
                        model_name=model_name)
            if np.mod(epoch, args.save_each)==0 or epoch==1 or epoch==args.epochs:

                sample_C = torch.randn(args.dim_C).to(device)
                sample_s = torch.randn(64, args.dim_s).to(device)
                samples = model._decode(sample_s, sample_C).cpu()

                save_image(samples, 'results/' + model_name + '/figs/samples/sample_' + str(epoch) + '.png')
                plt.close('all')

                save_model(model, epoch, model_name=model_name)