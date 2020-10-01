
import torch
from datasets import *
from models import *
import argparse
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from sklearn import metrics
from sklearn import mixture
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

########################################################################################################################
parser = argparse.ArgumentParser(description='Interpolation')
parser.add_argument('--dim_z', type=int, default=20, metavar='N',
                    help='Dimensions for local latent')
parser.add_argument('--dim_beta', type=int, default=50, metavar='N',
                    help='Dimensions for global latent')
parser.add_argument('--K', type=int, default=20, metavar='N',
                    help='Number of components for the Gaussian Global mixture')
parser.add_argument('--var_x', type=float, default=2e-1, metavar='N',
                    help='Number of components for the Gaussian Global mixture')
parser.add_argument('--arch', type=str, default='beta_vae',
                    help='Architecture for the model')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epoch', type=int, default=20,
                    help='Epoch to load')
parser.add_argument('--global_points', type=int, default=100, metavar='N',
                    help='Number of local points to plot')
parser.add_argument('--attribute', type=str, default='',
                    help='Attribute to plot (default None)')
parser.add_argument('--dim_reduction', type=str, default='tsne',
                    help='Dimensionality reduction to apply (default tsne)')
parser.add_argument('--model_name', type=str, default='UG-VAE/celeba',
                    help='name for the model to be saved')
args = parser.parse_args()



# load model

model = UGVAE(channels=nchannels['celeba'], dim_z=args.dim_z, dim_beta=args.dim_beta, K=args.K, arch=args.arch)
state_dict = torch.load('./results/' + args.model_name + '/checkpoints/checkpoint_' + str(args.epoch) + '.pth',
                        map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

ind = []
map = []

folder= 'results/' + args.model_name + '/figs/maps/' + 'epoch_' + str(args.epoch) + '/'
if os.path.isdir(folder) == False:
    os.makedirs(folder)


########################################################################################################################
# Encode celeba random batches
print('Encoding celebA random samples')
# load celeba dataset

celeba, _,  _ = get_data('celeba')
celeba_loader = torch.utils.data.DataLoader(celeba, batch_size = args.batch_size, shuffle=True)
celeba_iter = iter(celeba_loader)

for n in range(3*args.global_points):
    batch, label = celeba_iter.next()
    # Encode
    h = model.pre_encoder(batch)
    mu_z, var_z = model._encode_z(h)
    pi = model._encode_d(mu_z)
    mu_beta, var_beta = model._encode_beta(h, pi)
    map.append(mu_beta)
    ind.append(-1)



########################################################################################################################
# Encode celeba with atribute attr
print('Encoding celebA with selected attributes')
attrs = ['5_o_Clock_Shadow',
         'Arched_Eyebrows',
         'Attractive',
         'Bags_Under_Eyes',
         'Bald',
         'Bangs',
         'Big_Lips',
         'Big_Nose',
         'Black_Hair',
         'Blond_Hair',
         'Blurry',
         'Brown_Hair',
         'Bushy_Eyebrows',
         'Chubby',
         'Double_Chin',
         'Eyeglasses',
         'Goatee',
         'Gray_Hair',
         'Heavy_Makeup',
         'High_Cheekbones',
         'Male',
         'Mouth_Slightly_Open',
         'Mustache',
         'Narrow_Eyes',
         'No_Beard',
         'Oval_Face',
         'Pale_Skin',
         'Pointy_Nose',
         'Receding_Hairline',
         'Rosy_Cheeks',
         'Sideburns',
         'Smiling',
         'Straight_Hair',
         'Wavy_Hair',
         'Wearing_Earrings',
         'Wearing_Hat',
         'Wearing_Lipstick',
         'Wearing_Necklace',
         'Wearing_Necktie',
         'Young']

labels = ['Male', 'Female']
for i in range(2):
    # load celeba atributes dataset
    if i==0:
        celeba_attr, _,  _ = get_data('celeba_attribute', attr=20)
    else:
        celeba_attr, _, _ = get_data('celeba_nonattribute', attr=20)
    celeba_attr_loader = torch.utils.data.DataLoader(celeba_attr, batch_size=128, shuffle=True)
    celeba_attr_iter = iter(celeba_attr_loader)

    for n in range(args.global_points):
        batch, label = celeba_attr_iter.next()
        if n==0:
            save_image(batch.squeeze()[:5], folder + labels[i] + '.pdf', nrow=5)
        # Encode
        h = model.pre_encoder(batch.squeeze())
        mu_z, var_z = model._encode_z(h)
        pi = model._encode_d(mu_z)
        mu_beta, var_beta = model._encode_beta(h, pi)
        map.append(mu_beta)
        ind.append(i)

# MAP
map = torch.stack(map).detach().numpy()
ind=np.array(ind)

# t-SNE reduction
print('Training t-SNE...')
z = TSNE(n_components=2).fit_transform(map)

### Ploting
plt.figure(figsize=(6, 6))
#colors = ['b', 'r']
colors = [(16/256, 109/256, 165/256) , (124/256, 57/256, 187/256)]

# plot random
plt.plot(z[ind==-1, 0], z[ind==-1, 1], '.', color='k', alpha=0.7, label='random')
# plot attrs
[plt.plot(z[ind==i, 0], z[ind==i, 1], 'o', color=colors[i], alpha=0.7, label=labels[i]) for i in range(2)]

plt.legend(loc='best')
plt.savefig(folder + 'exp31.pdf')


########################################################################################################################
# Train a classifier over black and blond haired people, using global latent space

clf_lin = make_pipeline(StandardScaler(), SVC(kernel='linear', gamma='auto'))
clf_nolin = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='auto'))
X = map[3*args.global_points:]
y = ind[3*args.global_points:]
X = map
y = ind
y+=1
y[y==-1]=0

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
print('Fitting linear classifier')
clf_lin.fit(X_train, y_train)
print('Fitting non-linear classifier')
clf_nolin.fit(X_train, y_train)

print('Train accuracy on linear SVM: ' + str(100*clf_lin.score(X_train, y_train)))          # 100 in groups, 84.0 including random as class
print('Test accuracy on linear SVM: ' + str(100*clf_lin.score(X_test, y_test)))             # 85.0 in groups, 66.0 including random as class
print('Train accuracy on non-linear SVM: ' + str(100*clf_nolin.score(X_train, y_train)))    # 100.0 in groups, 89.0 including random as class
print('Test accuracy on non-linear SVM: ' + str(100*clf_nolin.score(X_test, y_test)))       # 85.0 in groups, 63.0 including random as class



