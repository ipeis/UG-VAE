from main import *
from torchvision.utils import make_grid
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


model = GLVAE().to(device)

name = 'z2_Z20'
epoch = 500     # Epoch to load
batch_size = 16    # N. images per sample
nsamples = 3    # N. samples per digit



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




########################################################################################################################
########################################################################################################################
#       Encode all samples in the dataset
########################################################################################################################
mnist = datasets.MNIST('../data', train=False, transform=transforms.ToTensor())
model.eval()

test_loader = torch.utils.data.DataLoader(
        mnist,
        batch_size=batch_size, **kwargs)

print('Encoding test set...')

mu_list = []
var_list = []
with torch.no_grad():
    for i, (data, _) in enumerate(test_loader):

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
# Encoding and save batches from each digit

print('Encoding batches with only each digit...')
mu_list = [[] for n in range(10)]
var_list = [[] for n in range(10)]

labels = []
Z_digits = []
aux = []
for n in range(10):
    mask = [1 if mnist[i][1] == n else 0 for i in range(len(mnist))]
    mask = torch.tensor(mask)
    aux.append(mask.sum())
    sampler = DigitSampler(mask, mnist)
    test_loader = torch.utils.data.DataLoader(
        mnist,
        sampler=sampler,
        batch_size=batch_size, **kwargs)

    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):

            recon_batch, mu, var, mu_g, var_g = model(data)
            labels.append(n)

            Z_digits.append(mu_g)
            if i < nsamples:

                sample = data.view(batch_size, 1, 28, 28)
                sample = make_grid(sample, nrow=int(np.sqrt(batch_size)), padding=2)
                save_image(sample, 'results/' + name + '/figs/Z' + str(n) + '_' + str(i) + '_' + str(epoch) + '.png')
                mu_list[n].append(mu_g)
                var_list[n].append(var_g)




########################################################################################################################
########################################################################################################################
# Build a t-SNE map with all the Zs

labels = np.array(labels)

Z_digits = torch.stack(Z_digits)
Z_samples = torch.cat([torch.stack(mu_list[n]) for n in range(10)])
Z_all = torch.cat([Z_digits, Z, Z_samples], dim=0)

Z_all = Z_all.numpy()

print('Performing t-SNE...')
tsne = TSNE(n_components=2, random_state=0)
X_all = tsne.fit_transform(Z_all)

X_digits = X_all[:len(Z_digits)]
X = X_all[len(Z_digits):]
X_samples = X_all[-len(Z_samples):]

fig, ax = plt.subplots(figsize=(12, 12))


plt.scatter(X_all[:, 0], X_all[:, 1], c='grey', label='mix')

colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple'
for i, c in enumerate(colors):
    plt.scatter(X_digits[labels == i, 0], X_digits[labels == i, 1], color=c, label=str(i))
plt.legend(loc='best')

plt.savefig('results/' + name + '/figs/global_posterior_' + str(epoch) + '_B' + str(batch_size) + '_colors.pdf')



########################################################################################################################
########################################################################################################################
# Plot in the map batches with the same digit

print('Building plot...')


for n in range(10):

    paths = ['results/' + name + '/figs/Z' + str(n) + '_' + str(i) + '_' + str(epoch) + '.png' for i in range(nsamples)]

    z = X_samples[n*nsamples:(n+1)*nsamples]

    for x0, y0, path in zip(z[:, 0], z[:, 1], paths):

        ab = AnnotationBbox(OffsetImage(plt.imread(path)), (x0, y0), frameon=False)
        ax.add_artist(ab)

plt.savefig('results/' + name + '/figs/global_posterior_' + str(epoch) + '_B' + str(batch_size) + '.pdf')















