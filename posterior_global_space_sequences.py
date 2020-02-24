from main import *
from torchvision.utils import make_grid
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


model = GLVAE().to(device)

name = '3_series_B200'
epoch = 500     # Epoch to load
batch_size = 128    # N. images per sample
nbatches = 200
nsamples = 3    # N. samples per digit
train_set = 'True'


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
#       Encode all samples in the dataset
########################################################################################################################
mnist = datasets.MNIST('../data', train=train_set, transform=transforms.ToTensor())
model.eval()

loader = torch.utils.data.DataLoader(
        mnist,
        batch_size=batch_size, **kwargs)

print('Encoding set...')

mu_list = []
var_list = []
with torch.no_grad():
    for i, (data, _) in enumerate(loader):

        #sample = data.view(batch_size, 1, 28, 28)
        #sample = make_grid(sample, nrow=2, padding=2)
        #save_image(sample, 'results/' + name + '/figs/Z' + str(n) + '_' + str(i) + '_' + str(epoch) + '.png')

        recon_batch, mu, var, mu_g, var_g = model(data)

        mu_list.append(mu_g)
        var_list.append(var_g)

Z = torch.stack(mu_list)
Z_var = torch.stack(var_list)




########################################################################################################################
########################################################################################################################
# Encoding and save batches from each series


mnist = datasets.MNIST('../data', train=train_set, transform=transforms.ToTensor())

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

########################################################################################

even = np.arange(args.batch_size) * 2
even = split_digits(even, args.batch_size)

odd = np.arange(args.batch_size) * 2 + 1
odd = split_digits(odd, args.batch_size)

fibonacci = [0, 1]
for i in range(2, args.batch_size):
    fibonacci.append(fibonacci[i-1]+fibonacci[i-2])
fibonacci = split_digits(fibonacci, args.batch_size)
primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103,
          107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
          227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283]
primes = split_digits(primes, args.batch_size)

series = [
    {'name': 'even',
     'values': even},
    {'name': 'odd',
     'values': odd},
    {'name': 'fibonacci',
     'values': fibonacci},
    {'name': 'primes',
     'values': primes},
]


########################################################################################################################
########################################################################################################################
#       Encode batches from series
########################################################################################################################

print('Encoding batches with only each series...')

iters = [iter(loaders[n]) for n in range(len(loaders))]
S = []
Z_series = []
with torch.no_grad():
    for batch_idx in range(nbatches):
        # Choose a serie
        s = np.random.randint(0, len(series))
        # Build batch
        batch = []
        i_s = np.zeros(len(series), dtype=int) # index of next digit for each serie
        for i in range(batch_size):
            n = series[s]['values'][i_s[s]]   # digit
            i_s += 1
            x, _ = iters[n].next()
            batch.append(x)

        data = torch.cat(batch).view(-1, 28, 28)

        recon_batch, mu, var, mu_g, var_g = model(data)
        Z_series.append(mu_g)
        S.append(s)

Z_series = torch.stack(Z_series)





########################################################################################################################
########################################################################################################################
# Build a t-SNE map with all the Zs

labels = np.array(S)

Z_all = torch.cat([Z_series, Z], dim=0)

Z_all = Z_all.numpy()

print('Performing t-SNE...')
tsne = TSNE(n_components=2, random_state=0)
X_all = tsne.fit_transform(Z_all)

X_series = X_all[:len(Z_series)]
X = X_all[len(Z_series):]

fig, ax = plt.subplots(figsize=(6, 6))


plt.scatter(X_all[:, 0], X_all[:, 1], c='grey', label='mix')

colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple'
for i, c in enumerate(colors[:len(series)]):
    plt.scatter(X_series[labels == i, 0], X_series[labels == i, 1], color=c, label=series[i]['name'])
plt.legend(loc='best')

if train_set==True:
    save_name = 'results/' + name + '/figs/global_posterior_sequences_' + 'train_' + str(epoch) + '_B' + str(batch_size) + '_colors.pdf'
else:
    save_name = 'results/' + name + '/figs/global_posterior_sequences_' + 'test_' + str(epoch) + '_B' + str(
        batch_size) + '_colors.pdf'

plt.savefig(save_name)
















