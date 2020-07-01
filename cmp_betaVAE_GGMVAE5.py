
import matplotlib.pyplot as plt
import numpy as np



def plot_losses(tr_losses, test_losses, tr_recs, test_recs,
                tr_klzs, test_klzs,
                model_name='model', linestyle='-'):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    if np.mean(tr_losses[:10])>=0:
        plt.semilogy(tr_losses, color=colors[0], linestyle=linestyle, label='train_loss_' + model_name)
        #plt.semilogy(test_losses, color=colors[0], linestyle=':')
        plt.semilogy(tr_recs, color=colors[1], linestyle=linestyle, label='train_rec_' + model_name)
        #plt.semilogy(test_recs, color=colors[1], linestyle=':')
        plt.semilogy(tr_klzs, color=colors[2], linestyle=linestyle, label=r'$KL_z$_' + model_name)
        #plt.semilogy(test_klzs, color=colors[2], linestyle=':')

    else:
        plt.plot(tr_losses, color=colors[0], linestyle=linestyle, label='train_loss_' + model_name)
        #plt.plot(test_losses, color=colors[0], linestyle=':')
        plt.plot(tr_recs, color=colors[1], linestyle=linestyle, label='train_rec_' + model_name)
        #plt.plot(test_recs, color=colors[1], linestyle=':')
        plt.plot(tr_klzs, color=colors[2], linestyle=linestyle, label=r'$KL_z$_' + model_name)
        #plt.plot(test_klzs, color=colors[2], linestyle=':')





model_name = 'GGMVAE5/mnist_var2e-1'
losses = np.load('results/' + model_name + '/checkpoints/losses.npy', allow_pickle=True).tolist()

plot_losses(tr_losses=losses['tr_losses'],
            test_losses=losses['test_losses'],
            tr_recs=losses['tr_recs'],
            test_recs=losses['test_recs'],
            tr_klzs=losses['tr_klzs'],
            test_klzs=losses['test_klzs'],
            model_name=model_name, linestyle='-'
            )


model_name = 'betaVAE/mnist'
losses = np.load('results/' + model_name + '/checkpoints/losses.npy', allow_pickle=True).tolist()
losses['tr_klzs'] = losses.pop('tr_kl_ls')
losses['test_klzs'] = losses.pop('test_kl_ls')

plot_losses(tr_losses=losses['tr_losses'],
            test_losses=losses['test_losses'],
            tr_recs=losses['tr_recs'],
            test_recs=losses['test_recs'],
            tr_klzs=losses['tr_klzs'],
            test_klzs=losses['test_klzs'],
            model_name=model_name, linestyle=':'
            )

plt.grid()
plt.xlabel('epoch')
plt.ylabel('mean loss')
plt.legend(loc='best')
plt.title('betaVAE vs GGMVAE5')
plt.savefig('./figs/cmp_losses.pdf')
