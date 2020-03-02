from GLVAE import *
import argparse
from torchvision import datasets, transforms
from torch import nn, optim
import re, sys


########################################################################################
parser = argparse.ArgumentParser(description='GLVAE MNIST Example')
parser.add_argument('--dim_z', type=int, default=10, metavar='N',
                    help='Dimensions for local latent')
parser.add_argument('--dim_Z', type=int, default=10, metavar='N',
                    help='Dimensions for global latent')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--nbatches_train', type=int, default=200, metavar='N',
                    help='Number of training batches from series (default: 200)')
parser.add_argument('--nbatches_test', type=int, default=20, metavar='N',
                    help='Number of test batches from series (default: 20)')
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

device = torch.device("cuda" if args.cuda else "cpu")

if os.path.isdir('results/' + model_name + '/checkpoints/') == False:
    os.makedirs('results/' + model_name + '/checkpoints/')

    if os.path.isdir('results/' + model_name + '/figs/') == False:
        os.makedirs('results/' + model_name + '/figs/')

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}




########################################################################################################################
########################################################################################################################
########################################################################################################################

def train_sequences_epoch(model, epoch, batch_size, nbatches, train_loaders, train_loaders_2, series, optimizer, cuda=False, log_interval=10):

    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model.train()
    train_loss = 0
    train_rec = 0
    train_kl_l = 0
    train_kl_g = 0

    iters = [iter(train_loaders[n]) for n in range(len(train_loaders))]
    iters_2 = [iter(train_loaders_2[n]) for n in range(len(train_loaders_2))]
    i_s = np.zeros(len(series), dtype=int)
    for batch_idx in range(nbatches):
        # Choose a serie
        s = np.random.randint(0, len(series))
        # Build batch
        batch = []
        # index of next digit for each serie
        for i in range(batch_size):

            n = series[s]['values'][i_s[s]]   # digit
            i_s[s] = np.mod(i_s[s] + 1, batch_size)
            k = int(np.round(np.random.uniform(0, 1)))
            if k == 0:
                x, _ = iters[n].next()
            else:
                x, _ = iters_2[n].next()
            batch.append(x)

        data = torch.cat(batch).view(-1, 28, 28)


        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, var, mu_g, var_g = model.forward(data)
        loss, rec, kl_l, kl_g = loss_function(recon_batch, data, mu, var, mu_g, var_g)
        loss.backward()
        train_loss += loss.item()
        train_rec += rec.item()
        train_kl_l += kl_l.item()
        train_kl_g += kl_g.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), nbatches*len(data),
                       100. * batch_idx / nbatches,
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / (nbatches*batch_size)))

    return train_loss / (nbatches*batch_size), train_rec / (nbatches*batch_size), \
           train_kl_l / (nbatches*batch_size), train_kl_g / (nbatches*batch_size)


def test(model, epoch, batch_size, nbatches, test_loaders, test_loaders_2, series, cuda=False, model_name='model'):

    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    model.eval()

    test_loss = 0
    test_rec = 0
    test_kl_l = 0
    test_kl_g = 0

    with torch.no_grad():

        iters = [iter(test_loaders[n]) for n in range(len(test_loaders))]
        iters_2 = [iter(test_loaders_2[n]) for n in range(len(test_loaders_2))]

        i_s = np.zeros(len(series), dtype=int)  # index of next digit for each serie

        for batch_idx in range(nbatches):
            # Choose a serie
            s = np.random.randint(0, len(series))
            # Build batch
            batch = []
            for i in range(batch_size):
                n = series[s]['values'][i_s[s]]  # digit
                i_s[s] = np.mod(i_s[s] + 1, batch_size)
                k = int(np.random.normal(0, 1))
                if k == 0:
                    x, _ = iters[n].next()
                else:
                    x, _ = iters_2[n].next()
                batch.append(x)

            data = torch.cat(batch).view(-1, 28, 28)

            data = data.to(device)
            recon_batch, mu, var, mu_g, var_g = model.forward(data)
            loss, rec, kl_l, kl_g = loss_function(recon_batch, data, mu, var, mu_g, var_g)
            test_loss += loss.item()
            test_rec += rec.item()
            test_kl_l += kl_l.item()
            test_kl_g += kl_g.item()

            if batch_idx == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data.view(batch_size, 1, 28, 28)[:n],
                                        recon_batch.view(batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                           'results/' + model_name + '/figs/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= (nbatches*batch_size)
    test_rec /= (nbatches*batch_size)
    test_kl_l /= (nbatches*batch_size)
    test_kl_g /= (nbatches*batch_size)
    print('====> Test set loss: {:.4f}'.format(test_loss))

    return test_loss, test_rec, test_kl_l, test_kl_g


def plot_latent(model, epoch, test_loader, cuda=False, model_name='model'):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model.eval()
    test_loss = 0
    z = []
    labels = []
    with torch.no_grad():
        for i, (data, l) in enumerate(test_loader):
            data = data.to(device)
            labels.append(l.to(device))
            recon_batch, mu, var, mu_g, var_g = model.forward(data)
            z.append(model.reparameterize(mu, var))

    z = torch.cat(z).cpu().numpy()[:1000]
    labels = torch.cat(labels).cpu().numpy()[:1000]

    # print('Performing t-SNE...')

    # X = TSNE(n_components=2, random_state=0).fit_transform(z)

    X = z
    plt.close('all')
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple'
    for i, c in enumerate(colors):
        plt.scatter(X[labels == i, 0], X[labels == i, 1], c=c, label=str(i))

    plt.legend(loc='best')
    plt.savefig('results/' + model_name + '/figs/local_latent_epoch_' + str(epoch) + '.pdf')


def plot_global_latent(model, epoch, nsamples, nreps, nims, cuda=False, model_name='model'):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model.eval()

    sample_z = torch.randn(nsamples, model.dim_z).to(device)

    # mu_Z, var_Z = model.global_encode(sample_z)
    # samples_Z = torch.stack([model.reparameterize(mu_Z, var_Z) for n in range(nreps)])
    samples_Z = torch.randn(nreps, model.dim_Z).to(device)

    pick = np.random.randint(0, len(samples_Z), nims)
    for s, sample_Z in enumerate(samples_Z[pick]):
        sample = model.decode(sample_z, sample_Z).cpu()
        sample = sample.view(nsamples, 1, 28, 28)
        sample = make_grid(sample, nrow=2, padding=2)
        save_image(sample, 'results/' + model_name + '/figs/Z' + str(s) + '_' + str(epoch) + '.png')

    Z = samples_Z.detach().cpu().numpy()

    print('Performing t-SNE...')
    X = TSNE(n_components=2, random_state=0).fit_transform(Z)

    paths = ['results/' + model_name + '/figs/Z' + str(s) + '_' + str(epoch) + '.png' for s in range(nreps)]

    fig, ax = plt.subplots(figsize=(12, 12))
    plt.scatter(X[:, 0], X[:, 1])

    for x0, y0, path in zip(X[pick, 0], X[pick, 1], paths):
        ab = AnnotationBbox(OffsetImage(plt.imread(path)), (x0, y0), frameon=False)
        ax.add_artist(ab)

    plt.savefig('results/' + model_name + '/figs/global_latent_epoch_' + str(epoch) + '.pdf')


def plot_losses(tr_losses, test_losses, tr_recs, test_recs, tr_kl_ls, test_kl_ls, tr_kl_gs, test_kl_gs,
                model_name='model'):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    plt.figure()
    plt.semilogy(tr_losses, color=colors[0], label='train_loss')
    plt.semilogy(test_losses, color=colors[0], linestyle=':')
    plt.semilogy(tr_recs, color=colors[1], label='train_rec')
    plt.semilogy(test_recs, color=colors[1], linestyle=':')
    plt.semilogy(tr_kl_ls, color=colors[2], label='KL_loc')
    plt.semilogy(test_kl_ls, color=colors[2], linestyle=':')
    plt.semilogy(tr_kl_gs, color=colors[3], label='KL_glob')
    plt.semilogy(test_kl_gs, color=colors[3], linestyle=':')
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


class DigitSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, mask, data_source):
        self.mask = mask
        self.data_source = data_source

    def __iter__(self):
        return iter([i.item() for i in torch.nonzero(self.mask)])

    def __len__(self):
        return len(self.data_source)


def split_digits(int_list, length):

    aux = int_list.copy()
    int_list = []
    for n in aux:
        digits = list(map(int, str(n)))
        int_list+=digits
    int_list = int_list[:length]

    return int_list


########################################################################################################################
########################################################################################################################
########################################################################################################################




mnist_tr = datasets.MNIST('./data', train=True, transform=transforms.ToTensor())

train_loaders = []
samplers = []
ndigits = []
for n in range(10):
    mask = [1 if mnist_tr[i][1] == n else 0 for i in range(len(mnist_tr))]
    mask = torch.tensor(mask)
    ndigits.append(mask.sum())
    samplers.append(DigitSampler(mask, mnist_tr))

    train_loaders.append(torch.utils.data.DataLoader(
        mnist_tr,
        sampler=samplers[-1],
        batch_size=1, **kwargs)
    )

########################################################################################################################
mnist_test = datasets.MNIST('./data', train=False, transform=transforms.ToTensor())

test_loaders = []
samplers = []
ndigits = []
for n in range(10):
    mask = [1 if mnist_test[i][1] == n else 0 for i in range(len(mnist_test))]
    mask = torch.tensor(mask)
    ndigits.append(mask.sum())
    samplers.append(DigitSampler(mask, mnist_test))

    test_loaders.append(torch.utils.data.DataLoader(
        mnist_test,
        sampler=samplers[-1],
        batch_size=1, **kwargs)
    )

########################################################################################################################


svhn_tr = datasets.SVHN('./data', split="train",
                        transform=transforms.Compose([transforms.CenterCrop(28), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]),
                        )

train_loaders_svhn = []
samplers_svhn = []
ndigits_svhn = []
for n in range(10):
    mask = [1 if svhn_tr[i][1] == n else 0 for i in range(len(svhn_tr))]
    mask = torch.tensor(mask)
    ndigits_svhn.append(mask.sum())
    samplers_svhn.append(DigitSampler(mask, svhn_tr))

    train_loaders_svhn.append(torch.utils.data.DataLoader(
        svhn_tr,
        sampler=samplers_svhn[-1],
        batch_size=1, **kwargs)
    )


########################################################################################################################


svhn_test = datasets.SVHN('./data', split="test",
                          transform=transforms.Compose([transforms.CenterCrop(28), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]),
                          )

test_loaders_svhn = []
samplers_svhn = []
ndigits_svhn = []
for n in range(10):
    mask = [1 if svhn_test[i][1] == n else 0 for i in range(len(svhn_test))]
    mask = torch.tensor(mask)
    ndigits_svhn.append(mask.sum())
    samplers_svhn.append(DigitSampler(mask, svhn_test))

    test_loaders_svhn.append(torch.utils.data.DataLoader(
        svhn_test,
        sampler=samplers_svhn[-1],
        batch_size=1, **kwargs)
    )





even = np.arange(args.batch_size*args.nbatches_train) * 2
even = split_digits(even, args.batch_size*args.nbatches_train)

odd = np.arange(args.batch_size*args.nbatches_train) * 2 + 1
odd = split_digits(odd, args.batch_size*args.nbatches_train)

fibonacci = [0, 1]
for i in range(2, args.batch_size*args.nbatches_train):
    fibonacci.append(fibonacci[i-1]+fibonacci[i-2])
fibonacci = split_digits(fibonacci, args.batch_size*args.nbatches_train)
"""
primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103,
          107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
          227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283]
primes = split_digits(primes, args.batch_size)

def isPrime(n):
    # see http://www.noulakaz.net/weblog/2007/03/18/a-regular-expression-to-check-for-prime-numbers/
    return re.match(r'^1?$|^(11+?)\1+$', '1' * n) == None


N = args.batch_size*args.nbatches_train # number of primes wanted (from command-line)
M = 100              # upper-bound of search space
primes = list()           # result list

while len(primes) < N:
    primes += filter(isPrime, range(M - 100, M)) # append prime element of [M - 100, M] to l
    M += 100                                # increment upper-bound

print(primes[:N]) # print result list limited to N elements
"""
series = [
    {'name': 'even',
     'values': even},
    {'name': 'odd',
     'values': odd},
    {'name': 'fibonacci',
     'values': fibonacci},
    #{'name': 'primes',
    # 'values': primes},
]

########################################################################################################################

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

        tr_loss, tr_rec, tr_kl_l, tr_kl_g = train_sequences_epoch(model, epoch, args.batch_size, args.nbatches_train, train_loaders, train_loaders_svhn, series,
                                                                  optimizer,
                                                                  cuda=args.cuda, log_interval=args.log_interval)
        tr_losses.append(tr_loss)
        tr_recs.append(tr_rec)
        tr_kl_ls.append(tr_kl_l)
        tr_kl_gs.append(tr_kl_g)

        test_loss, test_rec, test_kl_l, test_kl_g = test(model, epoch, args.batch_size, args.nbatches_test, test_loaders, test_loaders_svhn, series,
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
            plot_losses(tr_losses, test_losses, tr_recs, test_recs, tr_kl_ls, test_kl_ls, tr_kl_gs, test_kl_gs,
                    model_name=model_name)
            if np.mod(epoch, 100)==0 or epoch==1 or epoch==args.epochs:
                save_model(model, epoch, model_name=model_name)
                #plot_latent(model, epoch, test_loaders, cuda=args.cuda, model_name=model_name)
                #plot_global_latent(model, epoch, nsamples=4, nreps=1000, nims=20, cuda=args.cuda, model_name=model_name)


