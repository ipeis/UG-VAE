

- In models.py there are 3 classes with UG-VAE, ML-VAE and betaVAE implemented.
    
    To train them, you can run the script train_*.py (* is the model name) with default args, or defining custom args.
    
    Trained models will be saved in 'results/checkpoints'
    
- In datasets.py all the data sources used for the paper are defined.

- Experiments are implemented in exp*.py .


*Notes: 
- celeba dataset need to be downloaded from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html and extracted in data/celeba/ .

- faces dataset need to be download from https://faces.dmi.unibas.ch/bfm/index.php?nav=1-2&id=downloads and extrated in data/celeba/