import sys
sys.path.append('..')
from datasets import *
from models import *
import argparse
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

#----------------------------------------------------------------------------------------------------------------------#
# Arguments
parser = argparse.ArgumentParser(description='Plot q(beta|x)')
parser.add_argument('--dim_z', type=int, default=10, metavar='N',
                    help='Dimensions for local latent z (default: 10)')
parser.add_argument('--dim_beta', type=int, default=20, metavar='N',
                    help='Dimensions for global latent beta (default: 20)')
parser.add_argument('--K', type=int, default=10, metavar='N',
                    help='Number of components for the Gaussian mixture (default: 10)')
parser.add_argument('--var_x', type=float, default=2e-1, metavar='N',
                    help='Variance of p(x|z,beta) (default: 2e-1)')
parser.add_argument('--arch', type=str, default='k_vae',
                    help='Architecture for the model (default: k_vae)')
parser.add_argument('--epoch', type=int, default=8,
                    help='Epoch to load (default: 8)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training (default: False)')
parser.add_argument('--local_points', type=int, default=1000, metavar='N',
                    help='Number of local points to plot (default: 10)')
parser.add_argument('--global_points', type=int, default=200, metavar='N',
                    help='Number of local points to plot (default: 200)')
parser.add_argument('--attribute', type=str, default='',
                    help='Attribute to plot (default: '')')
parser.add_argument('--dim_reduction', type=str, default='tsne',
                    help='Dimensionality reduction to apply (default tsne)')
parser.add_argument('--model_name', type=str, default='mnist',
                    help='name for the model to be saved (default: mnist)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

#----------------------------------------------------------------------------------------------------------------------#
# Load data
data_tr, _, data_test = get_data('mnist_series', path='../data/')
train_loader = torch.utils.data.DataLoader(data_tr, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=True)

# Load model
device = torch.device("cuda" if args.cuda else "cpu")
model = UGVAE(channels=nchannels['mnist_series'], dim_z=args.dim_z, K=args.K, dim_beta=args.dim_beta, arch=args.arch, device=device).to(device)
state_dict = torch.load('../results/' + args.model_name + '/checkpoints/checkpoint_' + str(args.epoch) + '.pth',
                        map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# Create subfolder in log dir
folder = '../results/' + args.model_name + '/figs/maps/epoch_' + str(args.epoch) + '/'
if os.path.isdir(folder) == False:
    os.makedirs(folder)

# list with points of the map
mu_l = []
var_l = []
mu_g = []
var_g = []
labels_l = []
pis = []
iterator = iter(train_loader)
for i in range(args.global_points):

    # Load batch
    batch, labels = iterator.next()
    labels_l.append(labels.view(-1, labels.shape[-1]))

    # Encode
    recon_batch, mu_z, var_z, mus_z, vars_z, mu_beta, var_beta, pi = model(batch)

    # Save encodings
    pis.append(pi)
    mu_l.append(mu_z)
    var_l.append(var_z)
    mu_g.append(mu_beta)
    var_g.append(var_beta)
mu_l = torch.cat(mu_l).detach().numpy()
mu_g = torch.stack(mu_g).detach().numpy()
var_g = torch.stack(var_g).view(-1, var_beta.shape[-1]).detach().numpy()

labels_g = [labels_l[n][0, 1] for n in range(len(labels_l[:args.global_points]))]
labels_g = torch.tensor(labels_g).detach().numpy()
labels_l = np.concatenate(labels_l)[:args.local_points]

#----------------------------------------------------------------------------------------------------------------------#
# Building the map
if args.dim_reduction == 'tsne':
    print('Performing t-SNE...')
    tsne_l = TSNE(n_components=2, random_state=0)
    X_l = tsne_l.fit_transform(mu_l[:args.local_points])
    tsne_g = TSNE(n_components=2, random_state=0)
    X_g = tsne_g.fit_transform(mu_g.reshape(-1, args.dim_beta))
elif args.dim_reduction == 'kpca':
    print('Performing KPCA...')
    kpca_l = KernelPCA(n_components=2, kernel='linear')
    X_l = kpca_l.fit_transform(mu_l[:args.local_points])
    kpca_g = KernelPCA(n_components=2, kernel='linear')
    X_g = kpca_g.fit_transform(mu_g)


#----------------------------------------------------------------------------------------------------------------------#
# Local map
fig, ax = plt.subplots(figsize=(5, 5))
series = data_tr.series
idxs = [np.where(labels_l[:, 0] == n) for n in range(10)]
[plt.scatter(X_l[idxs[n], 0], X_l[idxs[n], 1], alpha=0.8, label=str(n)) for n in range(10)]
plt.legend(loc='best')
plt.title('Local space')
plt.title('Local space')
plt.savefig(folder + 'local_space_' + str(args.epoch) + '_' + args.dim_reduction)


#----------------------------------------------------------------------------------------------------------------------#
# Global map

mu_g_series = mu_g.copy()
var_g_series = var_g.copy()
labels_g_series = labels_g.copy()

# Encode random mnist batches
data_tr, _, data_test = get_data('mnist', path='../data/')
loader = torch.utils.data.DataLoader(data_tr, batch_size=128, shuffle=True)
iterator = iter(loader)
mu_l = []
var_l = []
mu_g = []
var_g = []
labels_l = []
for i in range(2 * args.global_points):
    batch, labels = iterator.next()
    labels_l.append(labels.view(-1, labels.shape[-1]))
    recon_batch, mu_z, var_z, mus_z, vars_z, mu_beta, var_beta, pi = model(batch)
    mu_l.append(mu_z)
    var_l.append(var_z)
    mu_g.append(mu_beta)
    var_g.append(var_beta)
mu_l = torch.cat(mu_l).detach().numpy()
mu_g = torch.stack(mu_g).detach().numpy()
var_g_mix = torch.stack(var_g).view(-1, var_beta.shape[-1]).detach().numpy()

# Concatenate random mnist and series from mnist in mu_g_all
mu_g_all = np.concatenate((mu_g, mu_g_series), axis=0)

if args.dim_reduction == 'tsne':
    print('Performing t-SNE...')
    tsne_g = TSNE(n_components=2, random_state=0)
    X_g_all = tsne_g.fit_transform(mu_g_all)
elif args.dim_reduction == 'kpca':
    print('Performing KPCA...')
    kpca_g = KernelPCA(n_components=2, kernel='linear')
    X_g_all = kpca_g.fit_transform(mu_g_all)
X_g_mix = X_g_all[:len(mu_g)]
X_g_series = X_g_all[len(mu_g):]


# Plot global map
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple'
fig, ax = plt.subplots(figsize=(5, 5))
plt.plot(X_g_mix[:, 0], X_g_mix[:, 1], '.', color=(27/256, 46/256, 104/256), alpha=0.8, label='random')
idxs = [np.where(labels_g_series == s) for s in range(len(series))]
[plt.scatter(X_g_series[idxs[s], 0], X_g_series[idxs[s], 1],  alpha=0.7, label=series[s]['name']) for s in
 range(len(series))]
plt.gca().set_prop_cycle(None)
plt.title('Global space')
plt.legend(loc='best')
plt.savefig(folder + 'global_space_' + str(args.epoch) + '_' + args.dim_reduction + '.pdf')

print('Finished')