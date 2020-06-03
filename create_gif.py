
from PIL import Image
import glob
import numpy as np

# Create the frames

model = 'mix/mnist'
epochs = np.concatenate(([1], 10*np.arange(1, 5)))
#im_list = [model + "figs/posterior/global_space_" + str(e) + "_ops_tsne.pdf" for e in epochs]

#imgs = glob.glob("*.png")
imgs = glob.glob("results/" + model + "/figs/posterior/*ops*")
frames = []
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)

# Save into a GIF file that loops forever
frames[0].save('posterior.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=300, loop=0)