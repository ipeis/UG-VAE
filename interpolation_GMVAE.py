

from models import *
import argparse
from torchvision import datasets, transforms
from torch import nn, optim
from datasets import *
from torchvision.utils import save_image
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE




class Interpolation():

    def __init__(self, steps):
        self.steps = steps

    def map_interpolation(self, model, loader, loader_mix, reps = 2, folder='./', labels_str=None):

        points=100
        # mixed data
        z_map = []
        labels = []
        for n in range(reps):
            batch, l = iter(loader_mix).next()
            # Encode
            h = model.pre_encoder(batch)
            mu_z, var_z = model._z_encode(h)
            mu_w, var_w = model._w_encode(h)
            pi_ = model._encode_d(mu_z, mu_w)
            z_map.append(mu_z)
            labels.append(l)
        z_map=z_map[:points]
        """
        ########################################################################################################################
        # Train a classifier over domains, using global latent space

        clf_lin = make_pipeline(StandardScaler(), SVC(kernel='linear', gamma='auto'))
        clf_nolin = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='auto'))
        X = torch.stack(beta_map).detach().numpy()
        y = np.concatenate((labels))
        #X = torch.stack(beta_map[-reps:]).detach().numpy()
        #y = np.cat(labels[-reps:])
        

        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
        print('Fitting linear classifier')
        clf_lin.fit(X_train, y_train)
        print('Fitting non-linear classifier')
        clf_nolin.fit(X_train, y_train)

        print('Train accuracy on linear SVM: ' + str(100*clf_lin.score(X_train, y_train)))          #  in groups,  including random as class
        print('Test accuracy on linear SVM: ' + str(100*clf_lin.score(X_test, y_test)))             #  in groups,  including random as class
        print('Train accuracy on non-linear SVM: ' + str(100*clf_nolin.score(X_train, y_train)))    #  in groups,  including random as class
        print('Test accuracy on non-linear SVM: ' + str(100*clf_nolin.score(X_test, y_test)))       #  in groups,  including random as class
        """

        

        # encode two batches (to interpolate between two samples)
        batch_1, l = iter(loader).next()
        labels.append(l)
        batch_2, l = iter(loader).next()
        labels.append(l)

        folder = 'results/' + name + '/figs/interpolation/' + folder
        if os.path.isdir(folder) == False:
            os.makedirs(folder)

        # Encode
        h1 = model.pre_encoder(batch_1)
        mu_z1, var_z1 = model._z_encode(h1)
        mu_w1, var_w1 = model._w_encode(h1)
        pi_1 = model._encode_d(mu_z1, mu_w1)


        h2 = model.pre_encoder(batch_2)
        mu_z2, var_z2 = model._z_encode(h2)
        mu_w2, var_w2 = model._w_encode(h2)
        pi_2 = model._encode_d(mu_z2, mu_w2)

        # INTERPOLATION
        mu_z1 = mu_z1[0]
        mu_z2 = mu_z2[0].view(-1, mu_z2.shape[-1])
        mu_w1 = mu_w1[0]
        mu_w2 = mu_w2[0].view(-1, mu_w2.shape[-1])

        lambda_ = torch.linspace(0, 1, self.steps)
        z_int = [l * mu_z2 + (1-l) * mu_z1 for l in lambda_]
        z_map.append(torch.stack(z_int).squeeze())

        grid = []
        for s1 in range(self.steps):
            recon = model._decode(z_int[s1])
            grid.append(recon)
        grid = torch.cat(grid)
        #grid = grid.permute(1, 0, 2, 3, 4)
        #grid = grid.reshape(-1, grid.shape[2], grid.shape[3], grid.shape[4])
        save_image(grid.cpu(),
                   folder + 'interpolation.pdf', nrow=steps, padding=1)

        # MAP
        z_map = torch.cat(z_map)

        labels = torch.cat(labels).detach().numpy()[:points]

        print('Training t-SNE...')
        z_tsne = TSNE(n_components=2).fit_transform(z_map.detach().numpy())

        plt.figure(figsize=(6, 6))

        markers = ['s', '^']
        colors = [(52/256, 77/256, 155/256), (53/256, 160/256, 38/256)]

        
        i=0
        if labels_str != None:
            for l in np.unique(labels):
                ind = labels==l
                plt.plot(z_tsne[:points,:][ind, 0], z_tsne[:points,:][ind, 1], 'o', color=colors[l], marker=markers[l], alpha = 0.7, label=labels_str[l])
                i+=1
        else:
            plt.plot(z_tsne[:, 0], z_tsne[:, 1], 'o')
        plt.plot(z_tsne[-self.steps, 0], z_tsne[-self.steps, 1], 'ko')
        plt.plot(z_tsne[-1, 0], z_tsne[-1, 1], 'k>')
        plt.plot(z_tsne[-self.steps:, 0], z_tsne[-self.steps:, 1], 'k-o', label='Interpolation')
        plt.legend(loc='best', fontsize=12)
        plt.grid()
        plt.savefig(folder + 'interpolation_map.pdf')
        


    def sampling(self, model, folder='./'):

        folder = 'results/' + name + '/figs/interpolation/' + folder
        if os.path.isdir(folder) == False:
            os.makedirs(folder)

        for d in range(model.L):
            print('Component ' + str(d))
            # Sample
            # beta1 = torch.randn(dim_beta)
            #d = torch.randint(model.L, [1,1])
            w1 = torch.squeeze(torch.ones(dim_w, 1)) * 1
            w2 = torch.squeeze(torch.ones(dim_w, 1)) * -1

            lambda_ = torch.linspace(0, 1, self.steps)
            w_int = [l * w2 + (1-l) * w1 for l in lambda_]
            grid = []
            for s1 in range(self.steps):
                mus_z, vars_z = model._z_prior(w_int[s1])
                mu_z = torch.squeeze(mus_z[d])
                var_z = torch.squeeze(vars_z[d])
                z1 = torch.squeeze(torch.ones_like(mu_z)) * mu_z - 3
                z2 = torch.squeeze(torch.ones_like(mu_z)) * mu_z + 3
                z = torch.stack([l * z2 + (1 - l) * z1 for l in lambda_])
                """
                z = torch.stack(
                    [Normal(mu_z, torch.diag(var_z)).sample() for i in range(5)])
                z = torch.stack(
                    [mu_z for i in range(5)])
                """
                recon = model._decode(z)
                grid.append(recon)
            grid = torch.cat(grid)
            #grid = grid.permute(1, 0, 2, 3, 4)
            #grid = grid.reshape(-1, grid.shape[2], grid.shape[3], grid.shape[4])
            save_image(grid.cpu(),
                       folder + 'sampling_interpolation_' + str(d) + '.pdf', nrow=self.steps, padding=1)




########################################################################################################################
parser = argparse.ArgumentParser(description='Interpolation')
parser.add_argument('--dim_z', type=int, default=20, metavar='N',
                    help='dimensions for local latent')
parser.add_argument('--dim_w', type=int, default=20, metavar='N',
                    help='dimensions for local noise latent')
parser.add_argument('--K', type=int, default=2, metavar='N',
                    help='Number of components for the Gaussian mixture')
parser.add_argument('--var_x', type=float, default=2e-1, metavar='N',
                    help='Variance of the data')
parser.add_argument('--dataset', type=str, default='celeba_faces_batch',
                    help='Name of the dataset')
parser.add_argument('--arch', type=str, default='beta_vae',
                    help='Architecture for the model')
parser.add_argument('--steps', type=int, default=7, metavar='N',
                    help='Number steps in z variable')
parser.add_argument('--epoch', type=int, default=3,
                    help='Epoch to load')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--model_name', type=str, default='GMVAE/celeba_faces',
                    help='name for the model to be saved')
args = parser.parse_args()

steps = args.steps
epoch = args.epoch
name = args.model_name

if __name__ == "__main__":

    data_tr, _, data_test = get_data(args.dataset)

    train_loader = torch.utils.data.DataLoader(data_tr, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=True)

    data_mix, _, _ = get_data(args.dataset[:-6])
    loader_mix = torch.utils.data.DataLoader(data_mix, batch_size=args.batch_size, shuffle=True)

    model = GMVAE(channels=nchannels[args.dataset], dim_z=args.dim_z, dim_w=args.dim_w, K=args.K, arch=args.arch)

    state_dict = torch.load('./results/' + args.model_name + '/checkpoints/checkpoint_' + str(args.epoch) + '.pth',
                            map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    labels_str = ['CelebA', 'FACES']
    #Interpolation(steps=steps).encoding(model, batch_1, batch_2, 'epoch_' + str(epoch) + '/')
    #Interpolation(steps=args.steps).sampling(model, 'epoch_' + str(args.epoch) + '/')
    Interpolation(steps=steps).map_interpolation(model, train_loader, loader_mix, reps=2,  folder='epoch_' + str(epoch) + '/', labels_str=labels_str)
