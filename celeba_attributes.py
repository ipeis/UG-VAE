from datasets import *
from models import *
import argparse
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import os
from sklearn.manifold import TSNE


#----------------------------------------------------------------------------------------------------------------------#
# Arguments
parser = argparse.ArgumentParser(description='Interpolation')
parser.add_argument('--dim_z', type=int, default=20, metavar='N',
                    help='Dimensions for local latent z (default: 20)')
parser.add_argument('--dim_beta', type=int, default=50, metavar='N',
                    help='Dimensions for global latent beta (default: 20)')
parser.add_argument('--K', type=int, default=20, metavar='N',
                    help='Number of components for the Gaussian mixt    ure (default: 20)')
parser.add_argument('--var_x', type=float, default=2e-1, metavar='N',
                    help='Variance of p(x|z,beta) (default: 2e-1)')
parser.add_argument('--arch', type=str, default='beta_vae',
                    help='Architecture for the model (default: beta_vae)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size for training (default: 128)')
parser.add_argument('--epoch', type=int, default=10,
                    help='Epoch to load (default: 20)')
parser.add_argument('--global_points', type=int, default=100, metavar='N',
                    help='Number of local points to plot (default: 100)')
parser.add_argument('--attribute', type=str, default='',
                    help='Attribute to plot (default: '')')
parser.add_argument('--dim_reduction', type=str, default='tsne',
                    help='Dimensionality reduction to apply (default tsne)')
parser.add_argument('--model_name', type=str, default='celeba',
                    help='name for the model to be saved (default: celeba)')
args = parser.parse_args()


#----------------------------------------------------------------------------------------------------------------------#
# Create subfolder in log dir
folder = 'results/' + args.model_name + '/figs/maps/' + 'epoch_' + str(args.epoch) + '/'
if os.path.isdir(folder) == False:
    os.makedirs(folder)

#----------------------------------------------------------------------------------------------------------------------#
# Load model
model = UGVAE(channels=nchannels['celeba'], dim_z=args.dim_z, dim_beta=args.dim_beta, K=args.K, arch=args.arch)
state_dict = torch.load('./results/' + args.model_name + '/checkpoints/checkpoint_' + str(args.epoch) + '.pth',
                        map_location=torch.device('cpu'))
model.load_state_dict(state_dict)



#----------------------------------------------------------------------------------------------------------------------#
# Load celeba dataset
celeba, _,  _ = get_data('celeba')
celeba_loader = torch.utils.data.DataLoader(celeba, batch_size = args.batch_size, shuffle=True)
celeba_iter = iter(celeba_loader)


#----------------------------------------------------------------------------------------------------------------------#
# Encode celeba random batches
print('Encoding celebA random samples')
ind = []
map = []
for n in range(3*args.global_points):
    batch, label = celeba_iter.next()
    # Encode
    h = model.pre_encoder(batch)
    mu_z, var_z = model._encode_z(h)
    pi = model._encode_d(mu_z)
    mu_beta, var_beta = model._encode_beta(h, pi)
    map.append(mu_beta)
    ind.append(-1)  # -1 means 'random'

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
    # Male:
    if i==0:
        celeba_attr, _,  _ = get_data('celeba_attribute', attr=20)
    # Female
    else:
        celeba_attr, _, _ = get_data('celeba_nonattribute', attr=20)
    celeba_attr_loader = torch.utils.data.DataLoader(celeba_attr, batch_size=128, shuffle=True)
    iterator = iter(celeba_attr_loader)

    for n in range(args.global_points):
        batch, label = iterator.next()
        # Save first 5 images as example
        if n==0:
            save_image(batch.squeeze()[:5], folder + labels[i] + '.pdf', nrow=5)
        # Encode
        h = model.pre_encoder(batch.squeeze())
        mu_z, var_z = model._encode_z(h)
        pi = model._encode_d(mu_z)
        mu_beta, var_beta = model._encode_beta(h, pi)
        map.append(mu_beta)
        ind.append(i)


#----------------------------------------------------------------------------------------------------------------------#
# Build map
map = torch.stack(map).detach().numpy()
ind=np.array(ind)

# t-SNE reduction
print('Training t-SNE...')
z = TSNE(n_components=2).fit_transform(map)

plt.figure(figsize=(6, 6))
colors = [(16/256, 109/256, 165/256) , (124/256, 57/256, 187/256)]

# plot random
plt.plot(z[ind==-1, 0], z[ind==-1, 1], '.', color='k', alpha=0.7, label='random')
# plot attrs
[plt.plot(z[ind==i, 0], z[ind==i, 1], 'o', color=colors[i], alpha=0.7, label=labels[i]) for i in range(2)]
plt.legend(loc='best')
plt.savefig(folder + 'male_female.pdf')


#----------------------------------------------------------------------------------------------------------------------#
# Train a classifier over male and female, using global latent space

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



