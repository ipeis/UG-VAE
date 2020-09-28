

from torch import nn
from datasets import *
import os
from models import *
from torchvision.utils import save_image


class MNISTclsf(nn.Module):
    def __init__(self):
        super(MNISTclsf, self).__init__()
        self.logits = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout(),
            View((-1, 3 * 3 * 64)),
            nn.Linear(3 * 3 * 64, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 10)
        )
        self.predictor = nn.LogSoftmax(dim=1)

    def forward(self, x):
            logits = self.logits(x)
            pred = self.predictor(logits)
            return pred


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


def train_epoch(model, epoch, loader, optimizer, cuda=False):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, labels) in enumerate(loader):
        optimizer.zero_grad()
        probs = model(data)
        loss = error(probs, labels)
        loss.backward()
        optimizer.step()

        # Total correct predictions
        predicted = torch.max(probs.data, 1)[1]
        correct += (predicted == labels).sum()

        accuracy = float(correct * 100) / float(batch_size * (batch_idx + 1))

        if batch_idx % 50 == 0:
            print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                epoch, batch_idx * len(data), len(loader.dataset), 100. * batch_idx / len(loader),
                loss.item(), accuracy))

    return accuracy

def test(model, loader):

    correct = 0
    for data, labels in loader:
        probs = model(data)
        predicted = torch.max(probs,1)[1]
        correct += (predicted == labels).sum()

    accuracy = 100*float(correct) / (len(loader)*batch_size)
    print("Test accuracy:{:.3f}% ".format( accuracy ))
    return accuracy

def align_test(model, loader, aligner, beta_global):

    correct = 0
    for i, (data, labels) in enumerate(loader):

        # Encode
        h = aligner.pre_encoder(data)
        mu_z, var_z = aligner._encode_z(h)
        pi = aligner._encode_d(mu_z)
        d = torch.argmax(pi, dim=1)
        mu_beta, var_beta = aligner._encode_beta(h, pi)
        beta = aligner.reparameterize(mu_beta, var_beta)
        mus_z, vars_z = aligner._z_prior(beta)
        locals = torch.stack([mus_z[di] for di in d])

        # Decode
        mu_x = aligner._decode(mu_z, beta_global)

        if i<5:

            folder = 'results/GGMVAE5/' + model_name + '/figs/domain_alignment/'
            if os.path.isdir(folder) == False:
                os.makedirs(folder)

            comparison = torch.cat([data[:5],
                                    mu_x[:5]])
            save_image(comparison.cpu(),
                       folder + 'example_' + str(i) + '.pdf', nrow=5)
        probs = model(mu_x)
        predicted = torch.max(probs,1)[1]
        correct += (predicted == labels).sum()

    accuracy = 100*float(correct) / (len(loader)*batch_size)
    print("Test accuracy:{:.3f}% ".format( accuracy ))
    return accuracy



def save_model(model, epoch, model_name='model'):
    folder = 'results/' + model_name + '/checkpoints/'
    if os.path.isdir(folder) == False:
        os.makedirs(folder)

    torch.save(model.state_dict(), folder + '/checkpoint_' + str(epoch) + '.pth')



###################################
###             Args            ###
###################################
model_name='mnist_clean_corrupted_batch'     # Name of the model to load
epoch = 2                  # Which epoch to load the generative model
train_clsf=False            # Train classifier (True) or don't (False)
batch_size = 128            # Digits per batch
epochs = 5                  # Epochs to train the classifier




###################################
###          Load data          ###
###################################
mnist_tr, _, mnist_test = get_data('mnist')
_, _, dist_mnist = get_data('corrupted_mnist')

mnist_tr_loader = torch.utils.data.DataLoader(mnist_tr, batch_size = batch_size, shuffle=True)
mnist_test_loader = torch.utils.data.DataLoader(mnist_test, batch_size = batch_size, shuffle=True)
dist_mnist_loader = torch.utils.data.DataLoader(dist_mnist, batch_size = batch_size, shuffle=True)



###################################
###     Train model (or load)   ###
###################################
clsf = MNISTclsf()
if train_clsf==True:

    optimizer = torch.optim.Adam(clsf.parameters())#,lr=0.001, betas=(0.9,0.999))
    error = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_epoch(clsf, epoch+1, mnist_tr_loader, optimizer)
        test(clsf, mnist_test_loader)

    save_model(clsf, epoch+1, 'MNISTclsf')

else:
    state_dict = torch.load('results/MNISTclsf/checkpoints/checkpoint_' + str(epochs) + '.pth',
                            map_location=torch.device('cpu'))
    clsf.load_state_dict(state_dict)


######################################################################################
###                 Compute accuracy for testing MNIST corrupted                      ###
######################################################################################
dist_mnist_loader = torch.utils.data.DataLoader(dist_mnist, batch_size = batch_size, shuffle=True)
accuracy = test(clsf, mnist_test_loader)
print("Test accuracy for clean-MNIST using clean-MNIST classifier: {:.3f}% ".format( accuracy ))
# 97.221%

accuracy = test(clsf, dist_mnist_loader)
print("Test accuracy for corrupted-MNIST using clean-MNIST classifier: {:.3f}% ".format( accuracy ))
# 51.365%

######################################################################################
###                         Load global generative model                           ###
######################################################################################

state_dict = torch.load('results/GGMVAE5/' +  model_name + '/checkpoints/checkpoint_' + str(epoch) + '.pth',
                            map_location=torch.device('cpu'))
model = GGMVAE5(channels=1, dim_z=10, dim_beta=20, L=10, var_x=2e-1, arch='k_vae')
model.load_state_dict(state_dict)


######################################################################
###             Compute mean global code for MNIST                 ###
######################################################################

mnist_beta = []
for batch_idx, (data, labels) in enumerate(mnist_tr_loader):

    # Encode
    h = model.pre_encoder(data)
    mu_z, var_z = model._encode_z(h)
    pi = model._encode_d(mu_z)
    mu_beta, var_beta = model._encode_beta(h, pi)
    mnist_beta.append(mu_beta)

mnist_beta = torch.stack(mnist_beta).mean(dim=0)


######################################################################
###            Decode corrupted MNIST using MNIST global code      ###
######################################################################
accuracy = align_test(clsf, dist_mnist_loader, model, mnist_beta)
print("Test accuracy for corrupted-MNIST using clean-MNIST classifier: {:.3f}% ".format( accuracy ))

