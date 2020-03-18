from train_sequences_svhn import *
from torchvision.utils import make_grid
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import KernelPCA

model = GLVAE(20, 20).to(device)

name = 'CNN-GLVAE'
epoch = 2000  # Epoch to load
batch_size = 128    # N. images per sample
nbatches = 400
train_set = True


########################################################################################################################
########################################################################################################################
state_dict = torch.load('results/' + name + '/checkpoints/checkpoint_' + str(epoch) + '.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)


########################################################################################################################
########################################################################################################################
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
# Encoding and save batches from each series


mnist = datasets.MNIST('../data', train=train_set, transform=transforms.Compose([transforms.Lambda(lambda image: image.convert('RGB')),
                                                      transforms.ToTensor()]), download=True)

loaders = []
samplers = []
for n in range(10):
    mask = [1 if mnist[i][1] == n else 0 for i in range(len(mnist))]
    mask = torch.tensor(mask)
    samplers.append(DigitSampler(mask, mnist))

    loaders.append(torch.utils.data.DataLoader(
        mnist,
        sampler=samplers[-1],
        batch_size=1, **kwargs)
    )


########################################################################################################################

if train_set:
    split = 'train'
else:
    split = 'test'
svhn = datasets.SVHN('./data', split=split,
                        transform=transforms.Compose([transforms.CenterCrop(28), transforms.ToTensor()]), download=True)

loaders_svhn = []
samplers_svhn = []
ndigits_svhn = []
for n in range(10):
    mask = [1 if svhn[i][1] == n else 0 for i in range(len(svhn))]
    mask = torch.tensor(mask)
    ndigits_svhn.append(mask.sum())
    samplers_svhn.append(DigitSampler(mask, svhn))

    loaders_svhn.append(torch.utils.data.DataLoader(
        svhn,
        sampler=samplers_svhn[-1],
        batch_size=1, **kwargs)
    )




########################################################################################################################
########################################################################################################################
#       Encode all samples in the dataset
########################################################################################################################
mnist = datasets.MNIST('../data', train=train_set, transform=transforms.Compose([transforms.Lambda(lambda image: image.convert('RGB')),
                                                      transforms.ToTensor()]))
model.eval()

loader = torch.utils.data.DataLoader(
        mnist,
        batch_size=1, **kwargs)

svhn_loader = torch.utils.data.DataLoader(
        svhn,
        batch_size=1, **kwargs)
print('Encoding set...')

mu_list = []
var_list = []
mu_local = []
var_local = []
digits = []
k_list = []
with torch.no_grad():
    iter_1 = iter(loader)
    iter_2 = iter(svhn_loader)

    for batch_idx in range(nbatches):

        # Build batch
        batch = []
        # index of next digit for each serie
        for i in range(batch_size):
            k = int(np.round(np.random.uniform(0, 1)))
            k_list.append(k)
            if k == 0:
                x, l = iter_1.next()
            else:
                x, l = iter_2.next()

            batch.append(x)
            digits.append(l)

        data = torch.cat(batch).view(-1, 3, 28, 28)
        recon_batch, mu, var, mu_g, var_g = model(data)

        mu_list.append(mu_g)
        var_list.append(var_g)

        mu_local.append(mu)
        var_local.append(var)

Z = torch.stack(mu_list)
Z_var = torch.stack(var_list)

z = torch.cat(mu_local)
z_var = torch.stack(var_local)
digits = torch.stack(digits)


########################################################################################

geo_2 = [2^n for n in range(args.batch_size)]
geo_2 = split_digits(geo_2, args.batch_size)

geo_5 = [5^n for n in range(args.batch_size)]
geo_5 = split_digits(geo_5, args.batch_size)

geo_7 = [7^n for n in range(args.batch_size)]
geo_7 = split_digits(geo_7, args.batch_size)

series = [
    {'name': 'geo_2',
     'values': geo_2},
    {'name': 'geo_5',
     'values': geo_5},
    {'name': 'geo_5',
     'values': geo_5},
    {'name': 'geo_7',
     'values': geo_7},
]


########################################################################################################################
########################################################################################################################
#       Encode batches from series
########################################################################################################################

print('Encoding batches with only each series...')

iters = [iter(loaders[n]) for n in range(len(loaders))]
S = []
Z_series = []
iters = [iter(loaders[n]) for n in range(len(loaders))]
iters_2 = [iter(loaders_svhn[n]) for n in range(len(loaders_svhn))]
i_s = np.zeros(len(series), dtype=int)
for batch_idx in range(nbatches):
    # Choose a serie
    s = np.random.randint(0, len(series))
    # Build batch
    batch = []
    # index of next digit for each serie
    for i in range(batch_size):

        n = series[s]['values'][i_s[s]]   # digit
        i_s[s] = np.mod(i_s[s]+1, batch_size)
        k = int(np.round(np.random.uniform(0, 1)))
        if k == 0:
            x, _ = iters[n].next()
        else:
            x, _ = iters_2[n].next()
        batch.append(x)

    data = torch.cat(batch).view(-1, 3, 28, 28)

    recon_batch, mu, var, mu_g, var_g = model(data)
    Z_series.append(mu_g)
    S.append(s)

Z_series = torch.stack(Z_series)



########################################################################################################################
########################################################################################################################
# Build a t-SNE map with all the zs (local)
"""
print('Performing t-SNE...')
tsne = TSNE(n_components=2, random_state=0)
index = np.random.choice(z.shape[0], 1000, replace=False)
x = tsne.fit_transform(z[index])
"""
print('Performing KPCA...')
kpca = KernelPCA(n_components=2, kernel='linear')
index = np.random.choice(z.shape[0], 1000, replace=False)
x = kpca.fit_transform(z[index])

digits = digits[index].squeeze().numpy()
k_list = np.array(k_list)[index]

fig, ax = plt.subplots(figsize=(6, 6))
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple'
markers = 'o', '*'
for i, c in enumerate(colors):
    plt.scatter(x[np.logical_and(digits == i , k_list == 0), 0], x[np.logical_and(digits == i , k_list == 0), 1], color=c, marker=markers[0], label='MNIST_' + str(i))
    plt.scatter(x[np.logical_and(digits == i , k_list == 1), 0], x[np.logical_and(digits == i , k_list == 1), 1], color=c, marker=markers[1], label='SVHN_' + str(i))
plt.title('Local latent space')
plt.legend(loc='best', ncol=2)

if train_set==True:
    save_name = 'results/' + name + '/figs/local_posterior_sequences_' + 'train_' + str(epoch) + '_B' + str(batch_size) + '_colors.pdf'
else:
    save_name = 'results/' + name + '/figs/local_posterior_sequences_' + 'test_' + str(epoch) + '_B' + str(
        batch_size) + '_colors.pdf'

plt.savefig(save_name)



########################################################################################################################
########################################################################################################################
# Build a t-SNE map with all the Zs

labels = np.array(S)

Z_all = torch.cat([Z_series, Z], dim=0)

Z_all = Z_all.detach().numpy()
"""
print('Performing t-SNE...')
tsne = TSNE(n_components=2, random_state=0)
X_all = tsne.fit_transform(Z_all)
"""
print('Performing KPCA...')
kpca = KernelPCA(n_components=2, kernel='linear')
X_all = kpca.fit_transform(Z_all)
X_series = X_all[:len(Z_series)]
X = X_all[len(Z_series):]

fig, ax = plt.subplots(figsize=(6, 6))


plt.scatter(X_all[:, 0], X_all[:, 1], c='grey', label='mix')

colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple'
for i, c in enumerate(colors[:len(series)]):
    plt.scatter(X_series[labels == i, 0], X_series[labels == i, 1], color=c, label=series[i]['name'])
plt.legend(loc='best')
plt.title('Global latent space')

if train_set==True:
    save_name = 'results/' + name + '/figs/global_posterior_sequences_' + 'train_' + str(epoch) + '_B' + str(batch_size) + '_colors.pdf'
else:
    save_name = 'results/' + name + '/figs/global_posterior_sequences_' + 'test_' + str(epoch) + '_B' + str(
        batch_size) + '_colors.pdf'

plt.savefig(save_name)
















