


from main import *
from torchvision.utils import make_grid
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


model = GLVAE().to(device)

name = 'z2_Z2'
epoch = 100
batch_size=4
nsamples = 4
nims = 20
nreps = 1000

dim_z = 2
dim_Z = 2


################################################################################################12  ########################
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
model.eval()


sample_z = torch.randn(nsamples, dim_z).to(device)

#mu_Z, var_Z = model.global_encode(sample_z)
#samples_Z = torch.stack([model.reparameterize(mu_Z, var_Z) for n in range(nreps)])
samples_Z = torch.randn(nreps, dim_Z).to(device)

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

plt.savefig('results/' + name + '/figs/global_prior_' + str(epoch) + '.pdf')