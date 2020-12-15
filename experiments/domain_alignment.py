import sys
sys.path.append('..')
from interpolation import *

#----------------------------------------------------------------------------------------------------------------------#
# Arguments
parser = argparse.ArgumentParser(description='Interpolation in UG-VAE')
parser.add_argument('--dim_z', type=int, default=40, metavar='N',
                    help='Dimensions for local latent z (default: 40)')
parser.add_argument('--dim_beta', type=int, default=40, metavar='N',
                    help='Dimensions for global latent beta (default: 40)')
parser.add_argument('--K', type=int, default=40, metavar='N',
                    help='Number of components for the Gaussian mixture (default: 40)')
parser.add_argument('--var_x', type=float, default=2e-1, metavar='N',
                    help='Variance of p(x|z,beta) (default: 2e-1)')
parser.add_argument('--dataset', type=str, default='celeba_faces',
                    help='Name of the dataset (default: celeba_faces)')
parser.add_argument('--arch', type=str, default='beta_vae',
                    help='Architecture for the model (default: beta_vae)')
parser.add_argument('--steps', type=int, default=7, metavar='N',
                    help='Number steps in local space and global space (default: 7)')
parser.add_argument('--epoch', type=int, default=10,
                    help='Epoch to load (default: 10)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--model_name', type=str, default='celeba_faces',
                    help='name for the model to load and save figs (default: celeba_faces)')
args = parser.parse_args()



#----------------------------------------------------------------------------------------------------------------------#
# Main
if __name__ == "__main__":

    # Load data
    data, _, _ = get_data(args.dataset, path='../data/')
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True)

    # Load model
    model = UGVAE(channels=nchannels[args.dataset], dim_z=args.dim_z, dim_beta=args.dim_beta, K=args.K, arch=args.arch)
    state_dict = torch.load('../results/' + args.model_name + '/checkpoints/checkpoint_' + str(args.epoch) + '.pth',
                            map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    # Perform the interpolation
    Interpolation(steps=args.steps).map_interpolation(
        model, loader, reps=100,  folder='epoch_' + str(args.epoch) + '/', labels_str=['celeba', 'faces'])
