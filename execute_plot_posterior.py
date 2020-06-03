


#from models import *
#from plot_posterior_MGLVAE import *
import numpy as np
import os

epochs = np.concatenate(([1], 10*np.arange(1, 14)))
for epoch in epochs:
    os.system("python3 plot_posterior_MGLVAE.py --epoch " + str(epoch))