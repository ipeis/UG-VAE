import torch
from datasets import *
from models import *
import argparse
from sklearn.decomposition import KernelPCA

########################################################################################################################
parser = argparse.ArgumentParser(description='Plot q(z|x)')
parser.add_argument('--dim_z', type=int, default=20, metavar='N',
                    help='Dimensions for local latent')
parser.add_argument('--dim_beta', type=int, default=2, metavar='N',
                    help='Dimensions for global latent')
parser.add_argument('--dim_w', type=int, default=2, metavar='N',
                    help='Dimensions for global latent')
parser.add_argument('--K', type=int, default=10, metavar='N',
                    help='Number of components for the Gaussian Global mixture')
parser.add_argument('--dataset', type=str, default='celeba_faces_batch',
                    help='Name of the dataset')
parser.add_argument('--arch', type=str, default='beta_vae',
                    help='Architecture for the model')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epoch', type=int, default=300,
                    help='Epoch to load')
parser.add_argument('--local_points', type=int, default=1000, metavar='N',
                    help='Number of local points to plot')
parser.add_argument('--global_points', type=int, default=200, metavar='N',
                    help='Number of local points to plot')
parser.add_argument('--attribute', type=str, default='',
                    help='Attribute to plot (default None)')
parser.add_argument('--dim_reduction', type=str, default='tsne',
                    help='Dimensionality reduction to apply (default tsne)')
parser.add_argument('--model_name', type=str, default='GGMVAE/celeba_faces_group',
                    help='name for the model to be saved')
args = parser.parse_args()

if args.dataset == 'mnist_series' or args.dataset == 'mnist_svhn_series' or args.dataset == 'mnist_series2':
    batch_size = 1
else:
    batch_size = args.batch_size

########################################################################################################################
data_tr, _, data_test = get_data(args.dataset)

train_loader = torch.utils.data.DataLoader(data_tr, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True)

########################################################################################################################

model = GGMVAE(channels=nchannels[args.dataset], dim_z=args.dim_z, dim_beta=args.dim_beta, dim_w = args.dim_w,
               K=args.K, arch=args.arch)

state_dict = torch.load('results/' + args.model_name + '/checkpoints/checkpoint_' + str(args.epoch) + '.pth',
                        map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

iterator = iter(train_loader)

mu_z = []
var_z = []
mu_beta = []
var_beta = []
mu_w = []
var_w = []
pi = []
mus_beta_p = []
vars_beta_p = []
labels_i = []
for i in range(args.global_points):
    batch, labels = iterator.next()

    labels_i.append(labels.view(-1, labels.shape[-1]))

    recon_batch, mu_z_, var_z_, mu_beta_, var_beta_, mu_w_, var_w_, pi_, mus_beta_p_, vars_beta_p_ = model(batch)

    mu_z.append(mu_z_)
    var_z.append(var_z_)

    mu_beta.append(mu_beta_)
    var_beta.append(var_beta_)

    mu_w.append(mu_w_)
    var_w.append(var_w_)

    pi.append(pi_)

    mus_beta_p.append(mus_beta_p_)
    vars_beta_p.append(vars_beta_p_)

mu_z = torch.cat(mu_z).detach().numpy()
mu_beta = torch.stack(mu_beta).detach().numpy()
var_beta = torch.stack(var_beta).view(-1, var_beta_.shape[-1]).detach().numpy()

if args.dataset == 'mnist_series' or args.dataset == 'mnist_svhn_series' or args.dataset == 'mnist_series2':
    labels_g = [labels_i[n][0, 1] for n in range(len(labels_i[:args.global_points]))]
    labels_g = torch.tensor(labels_g).detach().numpy()
elif args.dataset == 'celeba_faces_batch':
    labels_g = torch.stack([labels_i[n][0] for n in range(args.global_points)])
    labels_g = labels_g.detach().numpy()

labels_i = np.concatenate(labels_i)[:args.local_points]

########################################################################################################################
if args.dim_reduction == 'tsne':
    print('Performing t-SNE...')
    tsne_z = TSNE(n_components=2, random_state=0)
    X_z = tsne_z.fit_transform(mu_z[:args.local_points])
    #tsne_beta = TSNE(n_components=2, random_state=0)
    #X_beta = tsne_beta.fit_transform(mu_beta)
    X_beta = mu_beta
elif args.dim_reduction == 'kpca':
    print('Performing KPCA...')
    kpca_z = KernelPCA(n_components=2, kernel='linear')
    X_z = kpca_z.fit_transform(mu_z[:args.local_points])
    #kpca_beta = KernelPCA(n_components=2, kernel='linear')
    #X_beta = kpca_beta.fit_transform(mu_beta)
    X_beta = mu_beta

folder = 'results/' + args.model_name + '/figs/posterior/'
if os.path.isdir(folder) == False:
    os.makedirs(folder)

if args.dataset == 'mnist':
    labels_i = labels_i.reshape(-1, 1)[:args.local_points]

    ########################################################################################################################
    fig, ax = plt.subplots(figsize=(6, 6))
    idxs = [np.where(labels_i[:, 0] == l)[0] for l in range(10)]
    [plt.scatter(X_z[idxs[l], 0], X_z[idxs[l], 1], alpha=0.5, label=str(l)) for l in range(10)]
    plt.legend(loc='best')
    plt.title('Local space')

    plt.savefig(folder + 'local_space_' + str(args.epoch) + '_' + args.dim_reduction)
    ########################################################################################################################

    ########################################################################################################################
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.scatter(X_beta[:, 0], X_beta[:, 1], alpha=0.5)

    mus_beta = []
    vars_beta = []
    for s in range(10):
        sample_w = torch.randn(args.dim_w).to('cpu')
        mus_beta_, vars_beta_ = model._beta_gen(sample_w)
        mus_beta.append(mus_beta_)
        vars_beta.append(vars_beta_)

    mus_beta = torch.mean(torch.stack(mus_beta), dim=0).detach().numpy()
    vars_beta = torch.mean(torch.stack(vars_beta), dim=0).detach().numpy()
    plt.plot(mus_beta[:, 0], mus_beta[:, 1], 'ko', label=r'$\mu_\beta$')


    #plt.legend(loc='best')
    plt.title('Global space')
    plt.savefig(folder + 'global_space_' + str(args.epoch) + '_' + args.dim_reduction)
    ########################################################################################################################


elif args.dataset == 'mnist_series':

    ########################################################################################################################
    fig, ax = plt.subplots(figsize=(5, 5))
    series = data_tr.series
    idxs = [np.where(labels_i[:, 0] == l) for l in range(10)]
    [plt.scatter(X_z[idxs[l], 0], X_z[idxs[l], 1], alpha=0.8, label=str(l)) for l in range(10)]
    plt.legend(loc='best')
    plt.title('Local space')

    plt.savefig(folder + 'local_space_series_' + str(args.epoch) + '_' + args.dim_reduction)
    ########################################################################################################################

    mu_beta_series = mu_beta.copy()
    var_beta_series = var_beta.copy()
    labels_g_series = labels_g.copy()

    # Encode mix of mnist
    data_tr, _, data_test = get_data('mnist')
    loader = torch.utils.data.DataLoader(data_tr, batch_size=batch_size, shuffle=True)
    iterator = iter(loader)
    mu_z = []
    var_z = []
    mu_beta = []
    var_beta = []
    labels_i = []
    for i in range(2 * args.global_points):
        batch, labels = iterator.next()

        labels_i.append(labels.view(-1, labels.shape[-1]))

        recon_batch, mu_z_, var_z_, mu_beta_, var_beta_, mu_w_, var_w_, pi_, mus_beta_p_, vars_beta_p_ = model(batch)

        mu_z.append(mu_z_)
        var_z.append(var_z_)

        mu_beta.append(mu_beta_)
        var_beta.append(var_beta_)

    mu_z = torch.cat(mu_z).detach().numpy()
    mu_beta = torch.stack(mu_beta).detach().numpy()

    mu_beta_all = np.concatenate((mu_beta, mu_beta_series), axis=0)

    ########################################################################################################################
    if args.dim_reduction == 'tsne':
        print('Performing t-SNE...')
        tsne_beta = TSNE(n_components=2, random_state=0)
        #X_beta_all = tsne_beta.fit_transform(mu_beta_all)
        X_beta_all = mu_beta_all
    elif args.dim_reduction == 'kpca':
        print('Performing KPCA...')
        kpca_beta = KernelPCA(n_components=2, kernel='linear')
        #X_beta_all = kpca_beta.fit_transform(mu_beta_all)
        X_beta_all = mu_beta_all
    X_beta_mix = X_beta_all[:len(mu_beta)]
    X_beta_series = X_beta_all[len(mu_beta):len(mu_beta)+len(mu_beta_series)]

    ########################################################################################################################
    import matplotlib.patches as mpatches
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple'

    fig, ax = plt.subplots(figsize=(5, 5))
    plt.scatter(X_beta_mix[:, 0], X_beta_mix[:, 1], color='grey', alpha=0.5, label='random')
    """
    X = X_g_mix
    # Y = X_g_series[idxs[s], 1]
    stds = np.sqrt(var_g_mix)
    # stds_y = var_g_series[idxs[s], 1]
    for x, std in zip(X, stds):
        el = mpatches.Ellipse((x[0], x[1]), std[0], std[1], color='grey', alpha=0.1, zorder=0)
        ax.add_artist(el)
    idxs = [np.where(labels_g_series == s) for s in range(len(series))]
    for s in range(len(series)):

        X = X_g_series[idxs[s][0], :]
        #Y = X_g_series[idxs[s], 1]
        stds = np.sqrt(var_g_series[idxs[s][0], :])
        #stds_y = var_g_series[idxs[s], 1]
        for x, std in zip(X, stds):
            el = mpatches.Ellipse((x[0], x[1]), std[0], std[1], color=colors[s], alpha=0.2, zorder=5)
            ax.add_artist(el)
    """
    idxs = [np.where(labels_g_series == s) for s in range(len(series))]
    [plt.scatter(X_beta_series[idxs[s], 0], X_beta_series[idxs[s], 1],  alpha=0.5, label=series[s]['name']) for s in
     range(len(series))]
    plt.gca().set_prop_cycle(None)

    mus_beta = []
    vars_beta = []
    for s in range(10):
        sample_w = torch.randn(args.dim_w).to('cpu')
        mus_beta_, vars_beta_ = model._beta_gen(sample_w)
        mus_beta.append(mus_beta_)
        vars_beta.append(vars_beta_)

    mus_beta = torch.mean(torch.stack(mus_beta), dim=0).detach().numpy()
    vars_beta = torch.mean(torch.stack(vars_beta), dim=0).detach().numpy()
    plt.plot(mus_beta[:, 0], mus_beta[:, 1], 'ko', label=r'$\mu_\beta$')



    stds = np.sqrt(vars_beta)

    X = mus_beta
    for x, std in zip(X, stds):
        el = mpatches.Ellipse((x[0], x[1]), std[0], std[1], edgecolor='k', linestyle=':', fill=False, zorder=10)
        ax.add_artist(el)
    ax.scatter(X[:, 0], X[:, 1], color='k', label=r'$p(\beta)$', zorder=10)

    plt.title('Global space')
    #plt.axis([4.2, 4.6, 4.2, 4.6])
    #plt.axis([-4.20, -3.75, -4.4, -3.9])
    #plt.axis([-3.2, -2.95, -3.15, -2.9])
    #plt.axis([-4.2, -3.8, -4.35, -4])
    #plt.axis([-4.35, -4, -4.6, -4.25]) # 10
    #plt.axis([-4.5, -3.7, -5.4, -4.2]) # 1
    #plt.axis([-0.5, 0.5, 2.4, 3])
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='grey', alpha = 0.1),
                       Line2D([0], [0], marker='o', color=colors[0], alpha=0.2),
                       Line2D([0], [0], marker='o', color=colors[1], alpha=0.2),
                       Line2D([0], [0], marker='o', color=colors[2], alpha=0.2),
                       Line2D([0], [0], marker='o', color=colors[3], alpha=0.2),
                       Line2D([0], [0], marker='o', color='k')
                       ]
    legend_strings = ['random', 'even', 'odd', 'fibonacci', 'primes', r'$p(\beta)$']
    plt.legend(legend_elements, legend_strings, loc='best')
    """
    plt.legend(loc='best')
    plt.savefig(folder + 'global_space_series_' + str(args.epoch) + '_ops' + '_' + args.dim_reduction + '.pdf')
    ########################################################################################################################

if args.dataset == 'celeba':
    labels_i = labels_i.reshape(-1, 1)[:args.local_points]

    ########################################################################################################################
    fig, ax = plt.subplots(figsize=(6, 6))
    #idxs = [np.where(labels_i[:, 0] == l)[0] for l in range(10)]
    #[plt.scatter(X_z[idxs[l], 0], X_z[idxs[l], 1], alpha=0.5, label=str(l)) for l in range(10)]
    plt.scatter(X_z[:, 0], X_z[:, 1], alpha=0.5)
    #plt.legend(loc='best')
    plt.title('Local space')

    plt.savefig(folder + 'local_space_' + str(args.epoch) + '_' + args.dim_reduction)
    ########################################################################################################################

    ########################################################################################################################
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.scatter(X_beta[:, 0], X_beta[:, 1], alpha=0.5)

    mus_beta = []
    vars_beta = []
    for s in range(10):
        sample_w = torch.randn(args.dim_w).to('cpu')
        mus_beta_, vars_beta_ = model._beta_gen(sample_w)
        mus_beta.append(mus_beta_)
        vars_beta.append(vars_beta_)

    mus_beta = torch.mean(torch.stack(mus_beta), dim=0).detach().numpy()
    vars_beta = torch.mean(torch.stack(vars_beta), dim=0).detach().numpy()
    plt.plot(mus_beta[:, 0], mus_beta[:, 1], 'ko', label=r'$\mu_\beta$')

    stds = np.sqrt(vars_beta)

    import matplotlib.patches as mpatches
    X = mus_beta
    for x, std in zip(X, stds):
        el = mpatches.Ellipse((x[0], x[1]), std[0], std[1], edgecolor='k', linestyle=':', fill=False, zorder=10)
        ax.add_artist(el)
    ax.scatter(X[:, 0], X[:, 1], color='k', label=r'$p(\beta)$', zorder=10)


    #plt.legend(loc='best')
    plt.title('Global space')
    plt.savefig(folder + 'global_space_' + str(args.epoch) + '_' + args.dim_reduction)
    ########################################################################################################################


if args.dataset == 'mnist_svhn':

    d = labels_i[:args.local_points, 1]
    digits = labels_i[:args.local_points, 0]
    markers = 'o', '*'
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple'
    lab_leg = ['mnist', 'svhn']
    ########################################################################################################################
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, c in enumerate(colors):
        plt.scatter(X_z[np.logical_and(digits == i, d == 0), 0], X_z[np.logical_and(digits == i, d == 0), 1],
                    color=c, marker=markers[0], label=str(i), alpha=0.7)
        plt.scatter(X_z[np.logical_and(digits == i, d == 1), 0], X_z[np.logical_and(digits == i, d == 1), 1],
                    color=c, marker=markers[1], alpha=0.7)
    #plt.scatter(X_z[:, 0], X_z[:, 1], alpha=0.5)
    plt.legend(loc='best')
    plt.title('Local space')

    plt.savefig(folder + 'local_space_' + str(args.epoch) + '_' + args.dim_reduction)
    ########################################################################################################################

    ########################################################################################################################
    fig, ax = plt.subplots(figsize=(6, 6))
    #labels_g = labels_i.reshape(-1, batch_size, 2)
    #idxs = [np.where(labels_g[:, 0] == l)[0] for l in range(2)]
    #[plt.scatter(X_beta[idxs[l], 0], X_beta[idxs[l], 1], alpha=0.5, label=lab_leg[l]) for l in range(2)]
    plt.scatter(X_beta[:, 0], X_beta[:, 1], alpha=0.5)

    mus_beta = []
    vars_beta = []
    for s in range(10):
        sample_w = torch.randn(args.dim_w).to('cpu')
        mus_beta_, vars_beta_ = model._beta_gen(sample_w)
        mus_beta.append(mus_beta_)
        vars_beta.append(vars_beta_)

    mus_beta = torch.mean(torch.stack(mus_beta), dim=0).detach().numpy()
    vars_beta = torch.mean(torch.stack(vars_beta), dim=0).detach().numpy()
    plt.plot(mus_beta[:, 0], mus_beta[:, 1], 'ko', label=r'$\mu_\beta$')

    stds = np.sqrt(vars_beta)

    import matplotlib.patches as mpatches

    X = mus_beta
    for x, std in zip(X, stds):
        el = mpatches.Ellipse((x[0], x[1]), std[0], std[1], edgecolor='k', linestyle=':', fill=False, zorder=10)
        ax.add_artist(el)
    ax.scatter(X[:, 0], X[:, 1], color='k', label=r'$p(\beta)$', zorder=10)

    #plt.legend(loc='best')
    plt.title('Global space')
    plt.savefig(folder + 'global_space_' + str(args.epoch) + '_' + args.dim_reduction)
    ########################################################################################################################



if args.dataset == 'celeba_faces':

    d = labels_i.reshape(-1, 1)[:args.local_points]

    lab_leg = ['celeba', 'faces']
    ########################################################################################################################
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.scatter(X_z[:, 0], X_z[:, 1], alpha=0.5)
    plt.legend(loc='best')
    plt.title('Local space')

    plt.savefig(folder + 'local_space_' + str(args.epoch) + '_' + args.dim_reduction)
    ########################################################################################################################

    ########################################################################################################################
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.scatter(X_beta[:, 0], X_beta[:, 1], alpha=0.5)

    mus_beta = []
    vars_beta = []
    for s in range(10):
        sample_w = torch.randn(args.dim_w).to('cpu')
        mus_beta_, vars_beta_ = model._beta_gen(sample_w)
        mus_beta.append(mus_beta_)
        vars_beta.append(vars_beta_)

    mus_beta = torch.mean(torch.stack(mus_beta), dim=0).detach().numpy()
    vars_beta = torch.mean(torch.stack(vars_beta), dim=0).detach().numpy()
    plt.plot(mus_beta[:, 0], mus_beta[:, 1], 'ko', label=r'$\mu_\beta$')


    #plt.legend(loc='best')
    plt.title('Global space')
    plt.savefig(folder + 'global_space_' + str(args.epoch) + '_' + args.dim_reduction)
    ########################################################################################################################


if args.dataset == 'celeba_faces_batch':

    d = labels_i.reshape(-1, 1)[:args.local_points]

    lab_leg = ['celeba', 'faces']
    ########################################################################################################################
    fig, ax = plt.subplots(figsize=(6, 6))
    idxs = [np.where(labels_i[:, 0] == l)[0] for l in range(2)]
    [plt.scatter(X_z[idxs[l], 0], X_z[idxs[l], 1], alpha=0.5, label=lab_leg[l]) for l in range(2)]
    plt.legend(loc='best')
    plt.title('Local space')

    plt.savefig(folder + 'local_space_' + str(args.epoch) + '_' + args.dim_reduction)
    ########################################################################################################################

    ########################################################################################################################
    fig, ax = plt.subplots(figsize=(6, 6))
    idxs = [np.where(labels_i[:, 0] == l)[0] for l in range(2)]
    [plt.scatter(X_beta[idxs[l], 0], X_beta[idxs[l], 1], alpha=0.5, label=lab_leg[l]) for l in range(2)]

    mus_beta = []
    vars_beta = []
    for s in range(10):
        sample_w = torch.randn(args.dim_w).to('cpu')
        mus_beta_, vars_beta_ = model._beta_gen(sample_w)
        mus_beta.append(mus_beta_)
        vars_beta.append(vars_beta_)

    mus_beta = torch.mean(torch.stack(mus_beta), dim=0).detach().numpy()
    vars_beta = torch.mean(torch.stack(vars_beta), dim=0).detach().numpy()
    plt.plot(mus_beta[:, 0], mus_beta[:, 1], 'ko', label=r'$\mu_\beta$')


    #plt.legend(loc='best')
    plt.title('Global space')
    plt.savefig(folder + 'global_space_' + str(args.epoch) + '_' + args.dim_reduction)
    ########################################################################################################################



print('Finished')