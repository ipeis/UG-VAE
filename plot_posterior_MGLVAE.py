import torch
from datasets import *
from models import *
import argparse
from sklearn.decomposition import KernelPCA

########################################################################################################################
parser = argparse.ArgumentParser(description='Plot q(z|x)')
parser.add_argument('--dim_z', type=int, default=10, metavar='N',
                    help='Dimensions for local latent')
parser.add_argument('--dim_beta', type=int, default=2, metavar='N',
                    help='Dimensions for global latent')
parser.add_argument('--K', type=int, default=10, metavar='N',
                    help='Number of components for the Gaussian Global mixture')
parser.add_argument('--dataset', type=str, default='mnist_series2',
                    help='Name of the dataset')
parser.add_argument('--arch', type=str, default='k_vae',
                    help='Architecture for the model')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epoch', type=int, default=1,
                    help='Epoch to load')
parser.add_argument('--local_points', type=int, default=1000, metavar='N',
                    help='Number of local points to plot')
parser.add_argument('--global_points', type=int, default=150, metavar='N',
                    help='Number of local points to plot')
parser.add_argument('--attribute', type=str, default='',
                    help='Attribute to plot (default None)')
parser.add_argument('--dim_reduction', type=str, default='tsne',
                    help='Dimensionality reduction to apply (default tsne)')
parser.add_argument('--model_name', type=str, default='mix/mnist',
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

model = MGLVAE(channels=nchannels[args.dataset], dim_z=args.dim_z, dim_beta=args.dim_beta, K=args.K, arch=args.arch)

state_dict = torch.load('results/' + args.model_name + '/checkpoints/checkpoint_' + str(args.epoch) + '.pth',
                        map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

iterator = iter(train_loader)

mu_l = []
var_l = []
mu_g = []
var_g = []
labels_l = []
for i in range(args.global_points):
    print(i)
    batch, labels = iterator.next()

    labels_l.append(labels.view(-1, labels.shape[-1]))

    recon_batch, mu, var, pk, mu_g_, var_g_, mus_beta, vars_beta = model(batch)

    mu_l.append(mu)
    var_l.append(var)

    mu_g.append(mu_g_)
    var_g.append(var_g_)

mu_l = torch.cat(mu_l).detach().numpy()
mu_g = torch.stack(mu_g).detach().numpy()
var_g = torch.stack(var_g).view(-1, var_g_.shape[-1]).detach().numpy()

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
    #tsne_g = TSNE(n_components=2, random_state=0)
    #X_g = tsne_g.fit_transform(mu_g)
    X_g = mu_g
elif args.dim_reduction == 'kpca':
    print('Performing KPCA...')
    kpca_l = KernelPCA(n_components=2, kernel='linear')
    X_l = kpca_l.fit_transform(mu_l[:args.local_points])
    kpca_g = KernelPCA(n_components=2, kernel='linear')
    X_g = kpca_g.fit_transform(mu_g)

folder = 'results/' + args.model_name + '/figs/posterior/'
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
    plt.scatter(X_g[:, 0], X_g[:, 1], alpha=0.5)
    plt.title('Global space')
    plt.savefig(folder + 'global_space_' + str(args.epoch) + '_' + args.attribute + '_' + args.dim_reduction)
    ########################################################################################################################

elif args.dataset == 'mnist':
    ########################################################################################################################
    fig, ax = plt.subplots(figsize=(6, 6))
    idxs = [np.where(labels_l[:, 0] == l) for l in range(10)]
    [plt.scatter(X_l[idxs[l], 0], X_l[idxs[l], 1], alpha=0.5, label=str(l)) for l in range(10)]
    plt.legend(loc='best')
    plt.title('Local space')

    plt.savefig(folder + 'local_space_' + str(args.epoch) + '_' + args.dim_reduction)
    ########################################################################################################################

    ########################################################################################################################
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.scatter(X_g[:, 0], X_g[:, 1], alpha=0.5)
    plt.legend(loc='best')
    plt.title('Global space')
    plt.savefig(folder + 'global_space_' + str(args.epoch) + '_' + args.dim_reduction)
    ########################################################################################################################

    """
    elif args.dataset == 'mnist_series':

        ########################################################################################################################
        fig, ax = plt.subplots(figsize=(5, 5))
        series = data_tr.series
        idxs = [np.where(labels_l[:, 0] == l) for l in range(10)]
        [plt.scatter(X_l[idxs[l], 0], X_l[idxs[l], 1], alpha=0.8, label=str(l)) for l in range(10)]
        plt.legend(loc='best')
        plt.title('Local space')

        plt.savefig(folder + 'local_space_' + str(args.epoch) + '_' + args.dim_reduction)
        ########################################################################################################################

        mu_g_series = mu_g.copy()
        labels_g_series = labels_g.copy()

        data_tr, _, data_test = get_data('mnist')
        loader = torch.utils.data.DataLoader(data_tr, batch_size=batch_size, shuffle=True)
        iterator = iter(loader)
        mu_l = []
        var_l = []
        mu_g = []
        var_g = []
        labels_l = []
        for i in range(2*args.global_points):
            batch, labels = iterator.next()

            labels_l.append(labels.view(-1, labels.shape[-1]))

            recon_batch, mu, var, mu_g_, var_g_ = model(batch)

            mu_l.append(mu)
            var_l.append(var)

            mu_g.append(mu_g_)
            var_g.append(var_g_)

        mu_l = torch.cat(mu_l).detach().numpy()
        mu_g = torch.stack(mu_g).detach().numpy()

        mu_g_all = np.concatenate((mu_g, mu_g_series), axis=0)

        ########################################################################################################################
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


        ########################################################################################################################
        fig, ax = plt.subplots(figsize=(5, 5))
        plt.scatter(X_g_mix[:, 0], X_g_mix[:, 1], color='grey', alpha=0.7, label='random')

        idxs = [np.where(labels_g_series == s) for s in range(len(series))]
        [plt.scatter(X_g_series[idxs[s], 0], X_g_series[idxs[s], 1], alpha=0.7, label=series[s]['name']) for s in range(len(series))]
        plt.legend(loc='best')
        plt.title('Global space')
        plt.savefig(folder + 'global_space_' + str(args.epoch) + '_' + args.dim_reduction)
        ########################################################################################################################
    """

elif args.dataset == 'mnist_series':

    ########################################################################################################################
    fig, ax = plt.subplots(figsize=(5, 5))
    series = data_tr.series
    idxs = [np.where(labels_l[:, 0] == l) for l in range(10)]
    [plt.scatter(X_l[idxs[l], 0], X_l[idxs[l], 1], alpha=0.8, label=str(l)) for l in range(10)]
    plt.legend(loc='best')
    plt.title('Local space')

    plt.savefig(folder + 'local_space_' + str(args.epoch) + '_' + args.dim_reduction)
    ########################################################################################################################

    mu_g_series = mu_g.copy()
    var_g_series = var_g.copy()
    labels_g_series = labels_g.copy()

    # Encode series + offset
    data_args = {
        'offset': 2,
        #'scale': 2
    }
    ########################################################################################################################
    data_tr_off, _, data_test_off = get_data(args.dataset, **data_args)

    train_loader_off = torch.utils.data.DataLoader(data_tr_off, batch_size=batch_size, shuffle=True)
    test_loader_off = torch.utils.data.DataLoader(data_test_off, batch_size=batch_size, shuffle=True)
    iterator = iter(train_loader_off)

    mu_l = []
    var_l = []
    mu_g = []
    var_g = []
    labels_l = []
    for i in range(args.global_points):
        batch, labels = iterator.next()

        labels_l.append(labels.view(-1, labels.shape[-1]))

        recon_batch, mu, var, pk, mu_g_, var_g_, mus_beta, vars_beta = model.forward(batch)

        mu_l.append(mu)
        var_l.append(var)

        mu_g.append(mu_g_)
        var_g.append(var_g_)

    mu_l = torch.cat(mu_l).detach().numpy()
    mu_g = torch.stack(mu_g).detach().numpy()
    var_g = torch.stack(var_g).detach().numpy()

    mu_g_off = mu_g.copy()
    var_g_off = var_g.copy()
    labels_g_off = labels_g.copy()

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

        recon_batch, mu, var, pk, mu_g_, var_g_, mus_beta, vars_beta = model.forward(batch)

        mu_l.append(mu)
        var_l.append(var)

        mu_g.append(mu_g_)
        var_g.append(var_g_)

    mu_l = torch.cat(mu_l).detach().numpy()
    mu_g = torch.stack(mu_g).detach().numpy()
    var_g_mix = torch.stack(var_g).view(-1, var_g_.shape[-1]).detach().numpy()

    mu_g_all = np.concatenate((mu_g, mu_g_series, mu_g_off), axis=0)

    ########################################################################################################################
    if args.dim_reduction == 'tsne':
        print('Performing t-SNE...')
        tsne_g = TSNE(n_components=2, random_state=0)
        #X_g_all = tsne_g.fit_transform(mu_g_all)
        X_g_all = mu_g_all
    elif args.dim_reduction == 'kpca':
        print('Performing KPCA...')
        #kpca_g = KernelPCA(n_components=2, kernel='linear')
        #X_g_all = kpca_g.fit_transform(mu_g_all)
        X_g_all = mu_g_all
    X_g_mix = X_g_all[:len(mu_g)]
    X_g_series = X_g_all[len(mu_g):len(mu_g) + len(mu_g_series)]
    X_g_off = X_g_all[len(mu_g) + len(mu_g_series):]

    ########################################################################################################################
    import matplotlib.patches as mpatches
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple'

    fig, ax = plt.subplots(figsize=(5, 5))
    plt.scatter(X_g_mix[:, 0], X_g_mix[:, 1], color='grey', alpha=0.5, label='random')
    """
    X = X_g_mix
    # Y = X_g_series[idxs[s], 1]
    stds = np.sqrt(var_g_mix)
    # stds_y = var_g_series[idxs[s], 1]
    for x, std in zip(X, stds):
        el = mpatches.Ellipse((x[0], x[1]), std[0], std[1], color='grey', alpha=0.1, zorder=0)
        ax.add_artist(el)
    
    for s in range(len(series)):
        idxs = [np.where(labels_g_series == s) for s in range(len(series))]
        X = X_g_series[idxs[s][0], :]
        #Y = X_g_series[idxs[s], 1]
        stds = np.sqrt(var_g_series[idxs[s][0], :])
        #stds_y = var_g_series[idxs[s], 1]
        for x, std in zip(X, stds):
            el = mpatches.Ellipse((x[0], x[1]), std[0], std[1], color=colors[s], alpha=0.2, zorder=5)
            ax.add_artist(el)
    """
    [plt.scatter(X_g_series[idxs[s], 0], X_g_series[idxs[s], 1],  alpha=0.5, label=series[s]['name']) for s in
     range(len(series))]
    plt.gca().set_prop_cycle(None)

    #idxs = [np.where(labels_g_off == s) for s in range(len(series))]
    #[plt.scatter(X_g_off[idxs[s], 0], X_g_off[idxs[s], 1], marker='*', alpha=0.5, zorder=8) for s in
    # range(len(series))]
    #plt.scatter(mus_beta.detach().numpy()[:, 0], mus_beta.detach().numpy()[:, 1], c='k', label=r'$p(\beta)$')

    stds = np.sqrt(vars_beta.detach().numpy())

    X = mus_beta.detach().numpy()
    for x, std in zip(X, stds):
        el = mpatches.Ellipse((x[0], x[1]), std[0], std[1], edgecolor='k', linestyle=':', fill=False, zorder=10)
        ax.add_artist(el)
    ax.scatter(X[:, 0], X[:, 1], color='k', label=r'$p(\beta)$', zorder=10)

    plt.title('Global space')
    #plt.axis([4.2, 4.6, 4.2, 4.6])
    #plt.axis([-4.20, -3.75, -4.4, -3.9])
    #plt.axis([-3.2, -2.95, -3.15, -2.9])
    plt.axis([-4.2, -3.8, -4.35, -4])
    plt.axis([-4.35, -4, -4.6, -4.25]) # 10
    plt.axis([-4.5, -3.7, -5.4, -4.2]) # 1
    #plt.axis([-4.5, -3.5, -4.8, -4.1])
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
    plt.savefig(folder + 'global_space_' + str(args.epoch) + '_ops' + '_' + args.dim_reduction + '.pdf')
    ########################################################################################################################

    """
elif args.dataset == 'mnist_series':

    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple'

    ########################################################################################################################
    fig, ax = plt.subplots(figsize=(5, 5))
    series = data_tr.series
    idxs = [np.where(labels_l[:, 0] == l) for l in range(10)]
    [plt.scatter(X_l[idxs[l], 0], X_l[idxs[l], 1], alpha=0.8, label=str(l)) for l in range(10)]
    plt.legend(loc='best')
    plt.title('Local space')

    plt.savefig(folder + 'local_space_' + str(args.epoch) + '_' + args.dim_reduction)
    ########################################################################################################################

    mu_g_series = mu_g.copy()
    labels_g_series = labels_g.copy()

    # Encode series + offset
    data_args = {
        'offset': 5,
        #'scale': 2
    }
    ########################################################################################################################
    data_tr_off, _, data_test_off = get_data(args.dataset, **data_args)

    train_loader_off = torch.utils.data.DataLoader(data_tr_off, batch_size=batch_size, shuffle=True)
    test_loader_off = torch.utils.data.DataLoader(data_test_off, batch_size=batch_size, shuffle=True)
    iterator = iter(train_loader_off)

    mu_l = []
    var_l = []
    mu_g = []
    var_g = []
    labels_l = []
    batch_all = []
    for i in range(args.global_points):
        batch, labels = iterator.next()

        labels_l.append(labels.view(-1, labels.shape[-1]))

        recon_batch, mu, var, pk, mu_g_, var_g_, mus_beta, vars_beta = model.forward(batch)

        mu_l.append(mu)
        var_l.append(var)

        mu_g.append(mu_g_)
        var_g.append(var_g_)
        batch_all.append(batch)

    mu_l = torch.cat(mu_l).detach().numpy()
    mu_g = torch.stack(mu_g).detach().numpy()
    var_g = torch.stack(var_g).detach().numpy()

    mu_g_off = mu_g.copy()
    var_g_off = var_g.copy()
    labels_g_off = labels_g.copy()

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

        recon_batch, mu, var, pk, mu_g_, var_g_, mus_beta, vars_beta = model.forward(batch)

        mu_l.append(mu)
        var_l.append(var)

        mu_g.append(mu_g_)
        var_g.append(var_g_)

    mu_l = torch.cat(mu_l).detach().numpy()
    mu_g = torch.stack(mu_g).detach().numpy()

    mu_g_all = np.concatenate((mu_g, mu_g_series, mu_g_off), axis=0)

    ########################################################################################################################
    if args.dim_reduction == 'tsne':
        print('Performing t-SNE...')
        tsne_g = TSNE(n_components=2, random_state=0)
        X_g_all = tsne_g.fit_transform(mu_g_all)
    elif args.dim_reduction == 'kpca':
        print('Performing KPCA...')
        #kpca_g = KernelPCA(n_components=2, kernel='linear')
        #X_g_all = kpca_g.fit_transform(mu_g_all)
        X_g_all = mu_g_all
    X_g_mix = X_g_all[:len(mu_g)]
    X_g_series = X_g_all[len(mu_g):len(mu_g) + len(mu_g_series)]
    X_g_off = X_g_all[len(mu_g) + len(mu_g_series):]

    ########################################################################################################################
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.scatter(X_g_mix[:, 0], X_g_mix[:, 1], color='grey', alpha=0.7, label='random')

    idxs = [np.where(labels_g_series == s) for s in range(len(series))]

    [plt.scatter(X_g_series[idxs[s], 0], X_g_series[idxs[s], 1], color=colors[s], alpha=0.7, label=series[s]['name']) for s in
     range(len(series))]
    plt.gca().set_prop_cycle(None)
    idxs = [np.where(labels_g_off == s) for s in range(len(series))]
    for s in range(len(series)):
        for i in range(len(idxs[s])):
            plt.scatter(X_g_off[idxs[s][i], 0], X_g_off[idxs[s][i], 1], marker='*', alpha=0.7, color=colors[s])

    #[plt.scatter(X_g_off[idxs[s], 0], X_g_off[idxs[s], 1], marker='*', alpha=0.7) for s in
    #range(len(series))]
    plt.legend(loc='best')
    plt.title('Global space')
    plt.savefig(folder + 'global_space_' + str(args.epoch) + '_ops' + '_' + args.dim_reduction + '.pdf')
    ########################################################################################################################

    """


elif args.dataset == 'mnist_series2':

    ########################################################################################################################
    fig, ax = plt.subplots(figsize=(5, 5))
    series = data_tr.series
    idxs = [np.where(labels_l[:, 0] == l) for l in range(10)]
    [plt.scatter(X_l[idxs[l], 0], X_l[idxs[l], 1], alpha=0.8, label=str(l)) for l in range(10)]
    plt.legend(loc='best')
    plt.title('Local space')

    plt.savefig(folder + 'local_space_' + str(args.epoch) + '_' + args.dim_reduction)
    ########################################################################################################################

    mu_g_series = mu_g.copy()
    var_g_series = var_g.copy()
    labels_g_series = labels_g.copy()

    """
    # Encode series + offset
    data_args = {
        #'offset': 2,
        'scale': 2
    }
    ########################################################################################################################
    data_tr_off, _, data_test_off = get_data(args.dataset, **data_args)

    train_loader_off = torch.utils.data.DataLoader(data_tr_off, batch_size=batch_size, shuffle=True)
    test_loader_off = torch.utils.data.DataLoader(data_test_off, batch_size=batch_size, shuffle=True)
    iterator = iter(train_loader_off)

    mu_l = []
    var_l = []
    mu_g = []
    var_g = []
    labels_l = []
    for i in range(args.global_points):
        batch, labels = iterator.next()

        labels_l.append(labels.view(-1, labels.shape[-1]))

        recon_batch, mu, var, pk, mu_g_, var_g_, mus_beta, vars_beta = model.forward(batch)

        mu_l.append(mu)
        var_l.append(var)

        mu_g.append(mu_g_)
        var_g.append(var_g_)

    mu_l = torch.cat(mu_l).detach().numpy()
    mu_g = torch.stack(mu_g).detach().numpy()
    var_g = torch.stack(var_g).detach().numpy()

    mu_g_off = mu_g.copy()
    var_g_off = var_g.copy()
    labels_g_off = labels_g.copy()
    """

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

        recon_batch, mu, var, pk, mu_g_, var_g_, mus_beta, vars_beta = model.forward(batch)

        mu_l.append(mu)
        var_l.append(var)

        mu_g.append(mu_g_)
        var_g.append(var_g_)

    mu_l = torch.cat(mu_l).detach().numpy()
    mu_g = torch.stack(mu_g).detach().numpy()
    var_g_mix = torch.stack(var_g).view(-1, var_g_.shape[-1]).detach().numpy()

    mu_g_all = np.concatenate((mu_g, mu_g_series), axis=0)

    ########################################################################################################################
    if args.dim_reduction == 'tsne':
        print('Performing t-SNE...')
        tsne_g = TSNE(n_components=2, random_state=0)
        # X_g_all = tsne_g.fit_transform(mu_g_all)
        X_g_all = mu_g_all
    elif args.dim_reduction == 'kpca':
        print('Performing KPCA...')
        # kpca_g = KernelPCA(n_components=2, kernel='linear')
        # X_g_all = kpca_g.fit_transform(mu_g_all)
        X_g_all = mu_g_all
    X_g_mix = X_g_all[:len(mu_g)]
    X_g_series = X_g_all[len(mu_g):len(mu_g) + len(mu_g_series)]
    #X_g_off = X_g_all[len(mu_g) + len(mu_g_series):]

    ########################################################################################################################
    import matplotlib.patches as mpatches

    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple'

    fig, ax = plt.subplots(figsize=(5, 5))
    plt.scatter(X_g_mix[:, 0], X_g_mix[:, 1], color='grey', alpha=0.5, label='random')
    """
    X = X_g_mix
    # Y = X_g_series[idxs[s], 1]
    stds = np.sqrt(var_g_mix)
    # stds_y = var_g_series[idxs[s], 1]
    for x, std in zip(X, stds):
        el = mpatches.Ellipse((x[0], x[1]), std[0], std[1], color='grey', alpha=0.1, zorder=0)
        ax.add_artist(el)

    for s in range(len(series)):
        idxs = [np.where(labels_g_series == s) for s in range(len(series))]
        X = X_g_series[idxs[s][0], :]
        #Y = X_g_series[idxs[s], 1]
        stds = np.sqrt(var_g_series[idxs[s][0], :])
        #stds_y = var_g_series[idxs[s], 1]
        for x, std in zip(X, stds):
            el = mpatches.Ellipse((x[0], x[1]), std[0], std[1], color=colors[s], alpha=0.2, zorder=5)
            ax.add_artist(el)
    """
    idxs = [np.where(labels_g_series == s) for s in range(len(series))]
    [plt.scatter(X_g_series[idxs[s], 0], X_g_series[idxs[s], 1], alpha=0.5, label=series[s]['name']) for s in
     range(len(series))]
    plt.gca().set_prop_cycle(None)

    #idxs = [np.where(labels_g_off == s) for s in range(len(series))]
    #[plt.scatter(X_g_off[idxs[s], 0], X_g_off[idxs[s], 1], marker='*', alpha=0.5, zorder=8) for s in
    # range(len(series))]
    # plt.scatter(mus_beta.detach().numpy()[:, 0], mus_beta.detach().numpy()[:, 1], c='k', label=r'$p(\beta)$')

    stds = np.sqrt(vars_beta.detach().numpy())

    X = mus_beta.detach().numpy()
    for x, std in zip(X, stds):
        el = mpatches.Ellipse((x[0], x[1]), std[0], std[1], edgecolor='k', linestyle=':', fill=False, zorder=10)
        ax.add_artist(el)
    ax.scatter(X[:, 0], X[:, 1], color='k', label=r'$p(\beta)$', zorder=10)

    plt.title('Global space')
    # plt.axis([4.2, 4.6, 4.2, 4.6])
    # plt.axis([-4.20, -3.75, -4.4, -3.9])
    # plt.axis([-3.2, -2.95, -3.15, -2.9])
    #plt.axis([-4.2, -3.8, -4.35, -4])
    plt.axis([-5, -3.5, -5.5, -4.5])
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
    plt.savefig(folder + '_series2_global_space_' + str(args.epoch) + '_ops' + '_' + args.dim_reduction + '.pdf')
    ########################################################################################################################


elif args.dataset == 'mnist_svhn_series':

    ########################################################################################################################
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple'
    markers = 'o', '*'
    k_list = labels_l[:, 1]
    digits = labels_l[:, 0]
    for i, c in enumerate(colors):
        plt.scatter(X_l[np.logical_and(digits == i, k_list == 0), 0], X_l[np.logical_and(digits == i, k_list == 0), 1],
                    color=c, marker=markers[0], label=str(i), alpha=0.7)
        plt.scatter(X_l[np.logical_and(digits == i, k_list == 1), 0], X_l[np.logical_and(digits == i, k_list == 1), 1],
                    color=c, marker=markers[1], alpha=0.7)
    plt.title('Local latent space')
    plt.legend(loc='best', fontsize=8)

    plt.savefig(folder + 'local_space_' + str(args.epoch) + '_' + args.dim_reduction)

    ########################################################################################################################

    mu_g_series = mu_g.copy()
    labels_g_series = labels_g.copy()
    labels_l_series = labels_l.copy()

    data_tr_, _, data_test = get_data('mnist_svhn')
    loader = torch.utils.data.DataLoader(data_tr_, batch_size=128, shuffle=True)
    iterator = iter(loader)
    mu_l = []
    var_l = []
    mu_g = []
    var_g = []
    labels_l = []
    for i in range(2 * args.global_points):
        batch, labels = iterator.next()

        labels_l.append(labels.view(-1, labels.shape[-1]))

        recon_batch, mu, var, pk, mu_g_, var_g_, mus_beta, vars_beta = model.forward(batch)

        mu_l.append(mu)
        var_l.append(var)

        mu_g.append(mu_g_)
        var_g.append(var_g_)

    mu_l = torch.cat(mu_l).detach().numpy()
    mu_g = torch.stack(mu_g).detach().numpy()

    labels_l = np.concatenate(labels_l)

    mu_g_all = np.concatenate((mu_g, mu_g_series), axis=0)

    ########################################################################################################################
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

    ########################################################################################################################
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.scatter(X_g_mix[:, 0], X_g_mix[:, 1], color='grey', alpha=0.7, label='random')

    idxs = [np.where(labels_g_series == s) for s in range(len(data_tr.series))]
    for i in range(len(data_tr.series)):
        plt.scatter(X_g_series[labels_g_series == i, 0], X_g_series[labels_g_series == i, 1], alpha=0.7,
                    label=data_tr.series[i]['name'])
    plt.legend(loc='best')
    plt.title('Global space')
    plt.savefig(folder + 'global_space_' + str(args.epoch) + '_' + args.dim_reduction)
    ########################################################################################################################


elif args.dataset == 'celeba_faces_batch':
    ########################################################################################################################
    fig, ax = plt.subplots(figsize=(6, 6))
    idxs = [np.where(labels_l[:, 0] == l) for l in range(2)]
    [plt.scatter(X_l[idxs[l], 0], X_l[idxs[l], 1], alpha=0.5, label=str(l)) for l in range(2)]
    plt.legend(['celeba', 'faces'], loc='best')
    plt.title('Local space')

    plt.savefig(folder + 'local_space_' + str(args.epoch) + '_' + args.dim_reduction)
    ########################################################################################################################

    ########################################################################################################################
    fig, ax = plt.subplots(figsize=(6, 6))
    idxs = [np.where(labels_g[:, 0] == l) for l in range(2)]
    [plt.scatter(X_g[idxs[l], 0], X_g[idxs[l], 1], alpha=0.5, label=str(l)) for l in range(2)]
    plt.legend(loc='best')
    plt.title('Global space')
    plt.savefig(folder + 'global_space_' + str(args.epoch) + '_' + args.dim_reduction)
    ########################################################################################################################


elif args.dataset == 'fashion_mnist':
    classes = [ 'T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat',
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle boot'
                ]
    ########################################################################################################################
    fig, ax = plt.subplots(figsize=(6, 6))
    idxs = [np.where(labels_l[:, 0] == l) for l in range(10)]
    [plt.scatter(X_l[idxs[l], 0], X_l[idxs[l], 1], alpha=0.5, label=classes[l]) for l in range(10)]
    plt.legend(loc='best')
    plt.title('Local space')

    plt.savefig(folder + 'local_space_' + str(args.epoch) + '_' + args.dim_reduction)
    ########################################################################################################################

    ########################################################################################################################
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.scatter(X_g[:, 0], X_g[:, 1], alpha=0.5)
    plt.legend(loc='best')
    plt.title('Global space')
    plt.savefig(folder + 'global_space_' + str(args.epoch) + '_' + args.dim_reduction)
    ########################################################################################################################


print('Finished')