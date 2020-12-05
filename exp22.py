

from models import *
import argparse
from torchvision import datasets, transforms
from torch import nn, optim
from datasets import *
from torchvision.utils import save_image
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn import mixture
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



class Interpolation():

    def __init__(self, steps):
        self.steps = steps

    def map_interpolation(self, model, loader, loaders_mix, reps = 100, folder='./', labels_str=None):


        # mixed data
        C_map = []
        labels = []
        for n in range(reps):
            batch, _ = iter(loader_mix).next()
            # Encode
            mu_x, mu_s, var_s, mu_C, var_C = model(batch)
            C_map.append(mu_C)
            labels.append(-1)


        # grouped data
        for n in range(reps):
            batch, l = iter(loader).next()
            # Encode
            mu_x, mu_s, var_s, mu_C, var_C = model(batch)

            C_map.append(mu_C)
            labels.append(l[0])

        """
        ########################################################################################################################
        # Train a classifier over domains, using global latent space

        clf_lin = make_pipeline(StandardScaler(), SVC(kernel='linear', gamma='auto'))
        clf_nolin = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='auto'))
        X = torch.stack(C_map).detach().numpy()
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
        labels.append(l[0])
        batch_2, l = iter(loader).next()
        labels.append(l[0])

        folder = 'results/' + name + '/figs/interpolation/' + folder
        if os.path.isdir(folder) == False:
            os.makedirs(folder)

        # Encode
        mu_x1, s1, var_s1, C1, var_C1 = model(batch_1)
        mu_x2, s2, var_s2, C2, var_C2 = model(batch_2)

        # Take the first images of each batch
        s1 = s1[0].unsqueeze(0)
        s2 = s2[0].unsqueeze(0)

        # INTERPOLATION
        lambda_ = torch.linspace(0, 1, self.steps)
        C_int = [l * C2 + (1 - l) * C1 for l in lambda_]
        s_int = [l * s2 + (1 - l) * s1 for l in lambda_]
        grid = []
        for i1 in range(self.steps):
            for i2 in range(self.steps):
                recon = model._decode(s_int[i2], C_int[i1])
                grid.append(recon)
        grid = torch.cat(grid)
        save_image(grid.cpu(),
                   folder + 'encoding_interpolation.pdf', nrow=self.steps, padding=1)

        """
        # MAP
        C_map += C_int
        C_map = torch.stack(C_map)

        print('Training t-SNE...')
        C_tsne = TSNE(n_components=2).fit_transform(C_map.detach().numpy())

        plt.figure(figsize=(6, 6))


        markers = ['s', '^']
        colors = [(52/256, 77/256, 155/256), (53/256, 160/256, 38/256)]

        plt.plot(C_tsne[-self.steps, 0], C_tsne[-self.steps, 1], 'ko')
        plt.plot(C_tsne[-1, 0], C_tsne[-1, 1], 'k>')
        plt.plot(C_tsne[-self.steps:, 0], C_tsne[-self.steps:, 1], 'k-o', label='Interpolation')
        plt.legend(loc='best', fontsize=12)
        plt.grid()
        plt.savefig(folder + 'interpolation_map.pdf')
        """



########################################################################################################################
parser = argparse.ArgumentParser(description='Interpolation')
parser.add_argument('--dim_s', type=int, default=10, metavar='N',
                    help='Dimensions for local latent')
parser.add_argument('--dim_C', type=int, default=20, metavar='N',
                    help='Dimensions for global latent')
parser.add_argument('--dataset', type=str, default='celeba_faces_batch',
                    help='Name of the dataset')
parser.add_argument('--arch', type=str, default='beta_vae',
                    help='Architecture for the model')
parser.add_argument('--steps', type=int, default=7, metavar='N',
                    help='Number steps in z variable')
parser.add_argument('--epoch', type=int, default=10,
                    help='Epoch to load')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--model_name', type=str, default='MLVAE/celeba_faces',
                    help='name for the model to be saved')
args = parser.parse_args()


name = args.model_name
epoch = args.epoch
steps = args.steps
dataset = args.dataset
dim_s = args.dim_s
dim_C = args.dim_C


if __name__ == "__main__":

    data, _, _ = get_data(dataset)
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True)

    data_mix, _, _ = get_data(dataset[:-6])
    loader_mix = torch.utils.data.DataLoader(data_mix, batch_size=args.batch_size, shuffle=True)

    model = MLVAE(channels=nchannels[args.dataset], dim_s=dim_s, dim_C=dim_C, arch=args.arch)
    state_dict = torch.load('./results/' + name + '/checkpoints/checkpoint_' + str(epoch) + '.pth',
                            map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    labels_str = [
        'mix, 50% CelebA',
        'mix, 60% CelebA',
        'mix, 70% CelebA',
        'mix, 80% CelebA',
        'mix, 90% CelebA',
        'celebA',
        'FACES'
    ]
    Interpolation(steps=steps).map_interpolation(model, loader, loader_mix, reps=100,  folder='epoch_' + str(epoch) + '/',
                                                 labels_str=['mix',
                                                             'celebA', 'FACES'])


