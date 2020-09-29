
import matplotlib.pyplot as plt
import numpy as np



def plot_losses(tr_losses, test_losses, tr_recs, test_recs,
                tr_klzs, test_klzs, tr_klds, test_klds, tr_klbetas, test_klbetas,
                model_name='model'):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    plt.figure()
    if np.mean(tr_losses[:10])>=0:
        plt.semilogy(tr_losses, color=colors[0], label='train_loss')
        plt.semilogy(test_losses, color=colors[0], linestyle=':')
        plt.semilogy(tr_recs, color=colors[1], label='train_rec')
        plt.semilogy(test_recs, color=colors[1], linestyle=':')
        plt.semilogy(tr_klzs, color=colors[2], label=r'$KL_z$')
        plt.semilogy(test_klzs, color=colors[2], linestyle=':')
        plt.semilogy(tr_klds, color=colors[4], label=r'$KL_d$')
        plt.semilogy(test_klds, color=colors[4], linestyle=':')
        plt.semilogy(tr_klbetas, color=colors[5], label=r'$KL_\beta$')
        plt.semilogy(test_klbetas, color=colors[5], linestyle=':')

    else:
        plt.plot(tr_losses, color=colors[0], label='train_loss')
        plt.plot(test_losses, color=colors[0], linestyle=':')
        plt.plot(tr_recs, color=colors[1], label='train_rec')
        plt.plot(test_recs, color=colors[1], linestyle=':')
        plt.plot(tr_klzs, color=colors[2], label=r'$KL_z$')
        plt.plot(test_klzs, color=colors[2], linestyle=':')
        plt.plot(tr_klds, color=colors[4], label=r'$KL_d$')
        plt.plot(test_klds, color=colors[4], linestyle=':')
        plt.plot(tr_klbetas, color=colors[5], label=r'$KL_\beta$')
        plt.plot(test_klbetas, color=colors[5], linestyle=':')


    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('mean loss')
    plt.legend(loc='best')
    plt.savefig('results/' + model_name + '/figs/losses.pdf')



model_name = 'UG-VAE/mnist_var2e-1'


losses = np.load('results/' + model_name + '/checkpoints/losses.npy', allow_pickle=True).tolist()


plot_losses(tr_losses=losses['tr_losses'],
            test_losses=losses['test_losses'],
            tr_recs=losses['tr_recs'],
            test_recs=losses['test_recs'],
            tr_klzs=losses['tr_klzs'],
            test_klzs=losses['test_klzs'],
            tr_klds=losses['tr_klds'],
            test_klds=losses['test_klds'],
            tr_klbetas=losses['tr_klbetas'],
            test_klbetas=losses['test_klbetas'],
            model_name=model_name
            )



