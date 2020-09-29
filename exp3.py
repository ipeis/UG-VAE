


import torch
from datasets import *
from models import *
import argparse
from sklearn.decomposition import KernelPCA


########################################################################################################################
parser = argparse.ArgumentParser(description='Interpolation')
parser.add_argument('--dim_z', type=int, default=20, metavar='N',
                    help='Dimensions for local latent')
parser.add_argument('--dim_beta', type=int, default=50, metavar='N',
                    help='Dimensions for global latent')
parser.add_argument('--L', type=int, default=20, metavar='N',
                    help='Number of components for the Gaussian Global mixture')
parser.add_argument('--var_x', type=float, default=2e-1, metavar='N',
                    help='Number of components for the Gaussian Global mixture')
parser.add_argument('--dataset', type=str, default='celeba_attributes',
                    help='Name of the dataset')
parser.add_argument('--arch', type=str, default='k_vae',
                    help='Architecture for the model')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epoch', type=int, default=8,
                    help='Epoch to load')
parser.add_argument('--local_points', type=int, default=1000, metavar='N',
                    help='Number of local points to plot')
parser.add_argument('--global_points', type=int, default=200, metavar='N',
                    help='Number of local points to plot')
parser.add_argument('--attribute', type=str, default='',
                    help='Attribute to plot (default None)')
parser.add_argument('--dim_reduction', type=str, default='tsne',
                    help='Dimensionality reduction to apply (default tsne)')
parser.add_argument('--model_name', type=str, default='UG-VAE/mnist_series',
                    help='name for the model to be saved')
args = parser.parse_args()



########################################################################################################################
data_tr, _, data_test = get_data(args.dataset)

loader = torch.utils.data.DataLoader(data_tr, batch_size=args.batch_size, shuffle=True)


########################################################################################################################
device = 'cpu'

model = GGMVAE5(channels=nchannels[args.dataset], dim_z=args.dim_z, L=args.L, dim_beta=args.dim_beta, arch=args.arch, device=device).to(device)

state_dict = torch.load('results/' + args.model_name + '/checkpoints/checkpoint_' + str(args.epoch) + '.pth',
                        map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

iterator = iter(loader)

mu_l = []
var_l = []
mu_g = []
var_g = []
labels_l = []
pis = []
for i in range(args.global_points):

    batch, labels = iterator.next()

    labels_l.append(labels.view(-1, labels.shape[-1]))

    recon_batch, mu_z, var_z, mus_z, vars_z, mu_beta, var_beta, pi = model(batch)

    pis.append(pi)

    mu_l.append(mu_z)
    var_l.append(var_z)

    mu_g.append(mu_beta)
    var_g.append(var_beta)

mu_l = torch.cat(mu_l).detach().numpy()
mu_g = torch.stack(mu_g).detach().numpy()
var_g = torch.stack(var_g).view(-1, var_beta.shape[-1]).detach().numpy()






if args.dataset == 'mnist_series' or args.dataset == 'mnist_svhn_series' or args.dataset == 'mnist_series2':
    labels_g = [labels_l[n][0, 1] for n in range(len(labels_l[:args.global_points]))]
    labels_g = torch.tensor(labels_g).detach().numpy()
elif args.dataset == 'celeba_faces_batch':
    labels_g = [labels_l[n][0] for n in range(len(labels_l[:args.global_points]))]
    labels_g = torch.tensor(labels_g).detach().numpy()
labels_l = np.concatenate(labels_l)[:args.local_points]

########################################################################################################################
if args.dim_reduction == 'tsne':
    print('Performing t-SNE...')
    tsne_l = TSNE(n_components=2, random_state=0)
    X_l = tsne_l.fit_transform(mu_l[:args.local_points])
    tsne_g = TSNE(n_components=2, random_state=0)
    X_g = tsne_g.fit_transform(mu_g.reshape(-1, args.dim_beta))
    #X_g = X_g.reshape(-1, args.K, args.dim_beta)
    #X_g = mu_g
elif args.dim_reduction == 'kpca':
    print('Performing KPCA...')
    kpca_l = KernelPCA(n_components=2, kernel='linear')
    X_l = kpca_l.fit_transform(mu_l[:args.local_points])
    kpca_g = KernelPCA(n_components=2, kernel='linear')
    X_g = kpca_g.fit_transform(mu_g)

folder = 'results/' + args.model_name + '/figs/posterior/epoch_' + str(args.epoch) + '/'
if os.path.isdir(folder) == False:
    os.makedirs(folder)

if args.dataset == 'celeba' or args.dataset == 'light_celeba':
    ########################################################################################################################
    fig, ax = plt.subplots(figsize=(6, 6))
    if args.attribute != '':
        dim = data_tr.dataset.get_label_index(args.attribute)
        idx1 = np.where(labels_l[:, dim] == 1)
        idx0 = np.where(labels_l[:, dim] == -1)
        plt.scatter(X_l[idx1, 0], X_l[idx1, 1], alpha=0.5, label=args.attribute + '=1')
        plt.scatter(X_l[idx0, 0], X_l[idx0, 1], alpha=0.5, label=args.attribute + '=-1')
        plt.legend(loc='best')


    else:
        plt.scatter(X_l[:, 0], X_l[:, 1], alpha=0.5)
    plt.title('Local space')

    plt.savefig(folder + 'local_space_' + str(args.epoch) + '_' + args.attribute + '_' + args.dim_reduction)
    ########################################################################################################################

    ########################################################################################################################
    fig, ax = plt.subplots(figsize=(6, 6))
    [plt.scatter(X_g[:, k, 0], X_g[:, k, 1], alpha=0.5) for k in range(model.K)]
    plt.title('Global space')
    plt.savefig(folder + 'global_space_' + str(args.epoch) + '_' + args.attribute + '_' + args.dim_reduction)
    ########################################################################################################################

elif args.dataset == 'celeba_atributes':

    ########################################################################################################################
    fig, ax = plt.subplots(figsize=(5, 5))

    attr=1
    idxs = np.where(labels_l[:, 0] == attr)

    [plt.scatter(X_l[idxs[n], 0], X_l[idxs[n], 1], alpha=0.8, label=str(n)) for n in range(10)]
    plt.legend(loc='best')
    plt.title('Local space')
    """stds = np.sqrt(vars_z.detach().numpy())
    X = mus_z.detach().numpy()
    for x, std in zip(X, stds):
        el = mpatches.Ellipse((x[0], x[1]), std[0], std[1], edgecolor='k', linestyle=':', fill=False, zorder=10)
        ax.add_artist(el)
    ax.scatter(X[:, 0], X[:, 1], color='k', label=r'$p(\beta)$', zorder=10)
    """
    plt.title('Local space')

    plt.savefig(folder + 'local_space_' + str(args.epoch) + '_' + args.dim_reduction)
    ########################################################################################################################

    mu_g_series = mu_g.copy()
    var_g_series = var_g.copy()
    labels_g_series = labels_g.copy()

    ########################################################################################################################

    # Encode mix of mnist
    data_tr, _, data_test = get_data('mnist')
    loader = torch.utils.data.DataLoader(data_tr, batch_size=batch_size, shuffle=True)
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

    mu_g_all = np.concatenate((mu_g, mu_g_series), axis=0)


elif args.dataset == 'mnist_series':

    import matplotlib.patches as mpatches
    ########################################################################################################################
    fig, ax = plt.subplots(figsize=(5, 5))
    series = data_tr.series
    idxs = [np.where(labels_l[:, 0] == n) for n in range(10)]
    [plt.scatter(X_l[idxs[n], 0], X_l[idxs[n], 1], alpha=0.8, label=str(n)) for n in range(10)]
    plt.legend(loc='best')
    plt.title('Local space')
    """stds = np.sqrt(vars_z.detach().numpy())
    X = mus_z.detach().numpy()
    for x, std in zip(X, stds):
        el = mpatches.Ellipse((x[0], x[1]), std[0], std[1], edgecolor='k', linestyle=':', fill=False, zorder=10)
        ax.add_artist(el)
    ax.scatter(X[:, 0], X[:, 1], color='k', label=r'$p(\beta)$', zorder=10)
    """
    plt.title('Local space')


    plt.savefig(folder + 'local_space_' + str(args.epoch) + '_' + args.dim_reduction)
    ########################################################################################################################

    mu_g_series = mu_g.copy()
    var_g_series = var_g.copy()
    labels_g_series = labels_g.copy()

    ########################################################################################################################

    # Encode mix of mnist
    data_tr, _, data_test = get_data('mnist')
    loader = torch.utils.data.DataLoader(data_tr, batch_size=batch_size, shuffle=True)
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

    mu_g_all = np.concatenate((mu_g, mu_g_series), axis=0)

    ########################################################################################################################
    if args.dim_reduction == 'tsne':
        print('Performing t-SNE...')
        tsne_g = TSNE(n_components=2, random_state=0)
        X_g_all = tsne_g.fit_transform(mu_g_all)
        #X_g_all = mu_g_all
    elif args.dim_reduction == 'kpca':
        print('Performing KPCA...')
        kpca_g = KernelPCA(n_components=2, kernel='linear')
        X_g_all = kpca_g.fit_transform(mu_g_all)
        #X_g_all = mu_g_all
    X_g_mix = X_g_all[:len(mu_g)]
    X_g_series = X_g_all[len(mu_g):]
    #X_g_off = X_g_all[len(mu_g) + len(mu_g_series):]

    ########################################################################################################################
    import matplotlib.patches as mpatches
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple'

    fig, ax = plt.subplots(figsize=(5, 5))
    plt.scatter(X_g_mix[:, 0], X_g_mix[:, 1], color='grey', alpha=0.5, label='random')


    idxs = [np.where(labels_g_series == s) for s in range(len(series))]

    [plt.scatter(X_g_series[idxs[s], 0], X_g_series[idxs[s], 1],  alpha=0.5, label=series[s]['name']) for s in
     range(len(series))]
    plt.gca().set_prop_cycle(None)

    plt.title('Global space')


    plt.legend(loc='best')
    plt.savefig(folder + 'global_space_' + str(args.epoch) + '_ops' + '_' + args.dim_reduction + '.pdf')
    ########################################################################################################################

print('Finished')