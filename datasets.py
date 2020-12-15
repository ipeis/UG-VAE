
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import pandas as pd
from torch.utils.data import Subset
import numpy as np
import torch
import re


#----------------------------------------------------------------------------------------------------------------------#

class CelebA(Dataset):
    """
    Torchvision dataset implementation of celeba
    CelebA -> http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    celeba/ folder must be inside the given path.
    """
    def __init__(self, path='./data/celeba/', transform=None):

        self.path = path
        self.celeba = datasets.ImageFolder(path, transform)
        self.labels = pd.read_csv(self.path + 'list_attr_celeba.txt', sep='\s+', engine='python', header=1).values
        self.labels_keys = pd.read_csv(self.path + 'list_attr_celeba.txt', sep='\s+', engine='python', header=1).keys().values
        tr_list = pd.read_csv(self.path + 'list_eval_partition.txt', header=None).values
        partition = np.array([int(i[0][-1]) for i in tr_list])
        self.tr_idx = np.where(partition == 0)[0]
        self.val_idx = np.where(partition == 1)[0]
        self.test_idx = np.where(partition == 2)[0]

    def __getitem__(self, index):
        image = self.celeba.__getitem__(index)[0]
        labels = self.labels[index]
        return image, labels

    def get_label_index(self, label):
        return np.where(self.labels_keys==label)[0][0]

    def get_data_tr(self):
        return Subset(self, self.tr_idx), Subset(self, self.val_idx), Subset(self, self.test_idx)


#----------------------------------------------------------------------------------------------------------------------#

class MaskSampler(torch.utils.data.sampler.Sampler):
    """
    For obtaining samples from a dataset that accomplish a mask
    """
    def __init__(self, mask, data_source):
        self.mask = mask
        self.data_source = data_source

    def __iter__(self):
        return iter([i.item() for i in torch.nonzero(self.mask)])

    def __len__(self):
        return len(self.data_source)


#----------------------------------------------------------------------------------------------------------------------#

class MnistSeries(Dataset):
    """
    This class build batches from MNIST digits that follow series between [even, odd, fibonacci, primes] numbers
    """
    def __init__(self, path='./data/', split='train', nbatches=200, batch_size=128, offset=0, scale=1):
        # offset and scale to apply transformations to the series
        if split=='train':
            self.nbatches = nbatches
        elif split=='test':
            self.nbatches = int(0.1*nbatches)
        self.batch_size = batch_size
        self.length = int(256)
        even = np.arange(self.length) * 2
        even = even*scale + offset
        even = self.split_digits(even)

        odd = np.arange(self.length) * 2 + 1
        odd = odd*scale + offset
        odd = self.split_digits(odd)

        fibonacci = [0, 1]
        for i in range(2, self.length):
            fibonacci.append(fibonacci[i - 1] + fibonacci[i - 2])
        fibonacci = np.array(fibonacci)*scale + offset
        fibonacci = self.split_digits(fibonacci)

        def isPrime(n):
            # see http://www.noulakaz.net/weblog/2007/03/18/a-regular-expression-to-check-for-prime-numbers/
            return re.match(r'^1?$|^(11+?)\1+$', '1' * n) == None

        N = self.length # number of primes wanted
        M = 100              # upper-bound of search space
        primes = list()           # result list

        while len(primes) < N:
            primes += filter(isPrime, range(M - 100, M)) # append prime element of [M - 100, M] to l
            M += 100                                # increment upper-bound

        np.array(primes) * scale + offset
        primes = self.split_digits(primes)

        self.series = [
            {'name': 'even',
             'values': even[:self.length]},
            {'name': 'odd',
             'values': odd[:self.length]},
            {'name': 'fibonacci',
             'values': fibonacci[:self.length]},
            {'name': 'primes',
            'values': primes[:self.length]}
           ]

        self.mnist = datasets.MNIST(path, train=split=='train', download=True,
                                 transform=transforms.ToTensor())
        self.loaders = []
        samplers = []
        ndigits = []
        for n in range(10):
            mask = [1 if self.mnist[i][1] == n else 0 for i in range(len(self.mnist))]
            mask = torch.tensor(mask)
            ndigits.append(mask.sum())
            samplers.append(MaskSampler(mask, self.mnist))

            self.loaders.append(torch.utils.data.DataLoader(
                self.mnist,
                sampler=samplers[-1],
                batch_size=1)
            )
        self.iters = [iter(self.loaders[n]) for n in range(len(self.loaders))]
        self.i_s = np.array([np.random.randint(0, len(self.series[s]['values'])) for s in range(len(self.series))], dtype=int)
        self.s = np.random.randint(0, len(self.series))
        self.offset = offset

    def split_digits(self, int_list):

        aux = int_list.copy()
        int_list = []
        for n in aux:
            digits = list(map(int, str(n)))
            int_list += digits
        int_list = int_list

        return int_list
    def __getitem__(self, index):

        # Choose a serie
        s = torch.tensor(np.random.randint(0, len(self.series)))
        batch = []
        labels = []
        self.i_s = np.array([np.random.randint(0, len(self.series[s]['values'])) for s in range(len(self.series))], dtype=int)
        for i in range(self.batch_size):
            index = np.mod(self.i_s[s], self.length)
            n = self.series[s]['values'][index]  # digit
            self.i_s[s] += 1
            image, label = self.iters[n].next()
            batch.append(image)
            labels.append(label)

        data = torch.cat(batch).view(-1, 28, 28)
        labels = torch.cat(labels).view(-1, 1)
        labels = torch.cat((labels, torch.ones([self.batch_size, 1], dtype=torch.long)*s), dim=1).view(-1, 2)

        return data, labels

    def __len__(self):
        return self.nbatches

    def reset(self):
        self.loaders = []
        samplers = []
        ndigits = []
        for n in range(10):
            mask = [1 if self.mnist[i][1] == n else 0 for i in range(len(self.mnist))]
            mask = torch.tensor(mask)
            ndigits.append(mask.sum())
            samplers.append(MaskSampler(mask, self.mnist))

            self.loaders.append(torch.utils.data.DataLoader(
                self.mnist,
                sampler=samplers[-1],
                batch_size=1)
            )
        self.iters = [iter(self.loaders[n]) for n in range(len(self.loaders))]
        self.i_s = np.zeros(len(self.series), dtype=int)


#----------------------------------------------------------------------------------------------------------------------#

class CelebAFaces(Dataset):
    """
    This class build mixed batches with images from CelebA and 3D Faces dataset.
    faces/ and celeba/ folders must be inside the given path.
    CelebA -> http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    3D FACES -> https://faces.dmi.unibas.ch/bfm/index.php?nav=1-1-1&id=scans
    """
    def __init__(self, path='./data/', split='train', p=0.5):
        # p: proportion of images from celebA in each batch

        self.path = path
        self.p = p
        self.celeba = datasets.ImageFolder(path + 'celeba/', transform=transforms.Compose([
                                                  transforms.Resize((64, 64)),
                                                  transforms.ToTensor(),]))
        self.faces = datasets.ImageFolder(path + 'faces/', transform=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(), ]))

        celeba_tr_list = pd.read_csv(self.path+'celeba/' + 'list_eval_partition.txt', header=None).values
        celeba_partition = np.array([int(i[0][-1]) for i in celeba_tr_list])
        self.celeba_tr_idx = np.where(celeba_partition == 0)[0]
        self.celeba_val_idx = np.where(celeba_partition == 1)[0]
        self.celeba_test_idx = np.where(celeba_partition == 2)[0]

        if split=='train':
            self.celeba = Subset(self.celeba, self.celeba_tr_idx)
            self.faces = Subset(self.faces, np.arange(256))
        elif split=='test':
            self.celeba = Subset(self.celeba, self.celeba_test_idx)
            self.faces = Subset(self.faces, np.arange(256, 265))
        elif split=='val':
            self.celeba = Subset(self.celeba, self.celeba_val_idx)
            self.faces = Subset(self.faces, np.arange(265, 270))

        self.celeba_loader = iter(torch.utils.data.DataLoader(self.celeba, batch_size=1, shuffle=True))
        self.faces_loader = iter(torch.utils.data.DataLoader(self.faces, batch_size=1, shuffle=True))

        self.nims = int(0.9 * (self.celeba.__len__() + self.faces.__len__()) )
        self.fcount = 0

    def __getitem__(self, index):

        k = int(np.round(np.random.uniform(0, 1)))
        if k <= self.p:
            k=0
        else:
            k=1
        if k == 0:
            image, label = self.celeba_loader.next()
        else:
            image, label = self.faces_loader.next()
            self.fcount += 1
            if self.fcount == self.faces.__len__():
                self.reset_faces()
                self.fcount=0

        return image.view(image.shape[-3], image.shape[-2], image.shape[-1]), k

    def __len__(self):
        return self.nims

    def reset(self):
        self.celeba_loader = iter(torch.utils.data.DataLoader(self.celeba, batch_size=1, shuffle=True))
        self.faces_loader = iter(torch.utils.data.DataLoader(self.faces, batch_size=1, shuffle=True))

    def reset_faces(self):
        self.faces_loader = iter(torch.utils.data.DataLoader(self.faces, batch_size=1, shuffle=True))


#----------------------------------------------------------------------------------------------------------------------#

class CelebAFacesBatch(Dataset):
    """
    This class randomly select batches from celebA and 3D FACES datasets.
    For each batch, all the images come from the same dataset.
    faces/ and celeba/ folders must be inside the given path.
    CelebA -> http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    3D FACES -> https://faces.dmi.unibas.ch/bfm/index.php?nav=1-1-1&id=scans
    """
    def __init__(self, path='./data/', split='train', batch_size=128):

        self.path = path
        self.celeba = datasets.ImageFolder(path + 'celeba/', transform=transforms.Compose([
                                                  transforms.Resize((64, 64)),
                                                  transforms.ToTensor(),]))
        self.faces = datasets.ImageFolder(path + 'faces/', transform=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(), ]))
        self.batch_size = batch_size
        celeba_tr_list = pd.read_csv(self.path+'celeba/' + 'list_eval_partition.txt', header=None).values
        celeba_partition = np.array([int(i[0][-1]) for i in celeba_tr_list])
        self.celeba_tr_idx = np.where(celeba_partition == 0)[0]
        self.celeba_val_idx = np.where(celeba_partition == 1)[0]
        self.celeba_test_idx = np.where(celeba_partition == 2)[0]

        if split=='train':
            self.celeba = Subset(self.celeba, self.celeba_tr_idx)
            self.faces = Subset(self.faces, np.arange(256))
        elif split=='test':
            self.celeba = Subset(self.celeba, self.celeba_test_idx)
            self.faces = Subset(self.faces, np.arange(256, 265))
        elif split=='val':
            self.celeba = Subset(self.celeba, self.celeba_val_idx)
            self.faces = Subset(self.faces, np.arange(265, 270))

        self.celeba_loader = iter(torch.utils.data.DataLoader(self.celeba, batch_size=1, shuffle=True))
        self.faces_loader = iter(torch.utils.data.DataLoader(self.faces, batch_size=1, shuffle=True))
        self.nims = int(0.8 * (self.celeba.__len__() + self.faces.__len__()) )
        self.count = 0
        self.k = int(np.round(np.random.uniform(0, 1)))
        self.fcount=0

    def __getitem__(self, index):
        if self.k == 0:
            image, label = self.celeba_loader.next()
        else:
            image, label = self.faces_loader.next()
            self.fcount += 1
            if self.fcount == self.faces.__len__():
                self.reset_faces()
                self.fcount=0

        self.count += 1
        if self.count == self.batch_size:
            self.count = 0
            self.k = int(np.round(np.random.uniform(0, 1))) # For random picks
            #self.k = 0 if self.k==1 else 1 # for switching after batch

        label = self.k

        return image.view(image.shape[-3], image.shape[-2], image.shape[-1]), label

    def __len__(self):
        return self.nims

    def reset(self):
        self.celeba_loader = iter(torch.utils.data.DataLoader(self.celeba, batch_size=1, shuffle=True))
        self.faces_loader = iter(torch.utils.data.DataLoader(self.faces, batch_size=1, shuffle=True))

    def reset_faces(self):
        self.faces_loader = iter(torch.utils.data.DataLoader(self.faces, batch_size=1, shuffle=True))


#----------------------------------------------------------------------------------------------------------------------#

class CelebAttribute(Dataset):
    """
    This class builds CelebA batches with a given attribute
    CelebA -> http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    celeba/ folder must be inside the given path.
    Attributes must be available in list_attr_celeba.txt inside the path
    """
    def __init__(self, path='./data/', attr = 0):

        self.celeba = datasets.ImageFolder(path, transform=transforms.Compose([
                                                  transforms.Resize((64, 64)),
                                                  transforms.ToTensor(),]))

        self.labels = pd.read_csv(path + 'list_attr_celeba.txt', sep='\s+', engine='python', header=1).values
        self.labels_keys = pd.read_csv(path + 'list_attr_celeba.txt', sep='\s+', engine='python',
                                       header=1).keys().values

        mask = torch.tensor( [1 if self.labels[i][attr] == 1 else 0 for i in range(len(self.celeba))] )
        self.nsamples = mask.sum()
        self.sampler = MaskSampler(mask, self.celeba)
        self.loader = torch.utils.data.DataLoader(
                self.celeba,
                sampler=self.sampler,
                batch_size=1)
        self.iter = iter(self.loader)


    def __getitem__(self, index):

        image, label = self.iter.next()

        return image, label

    def __len__(self):
        return self.nsamples


#----------------------------------------------------------------------------------------------------------------------#

class CelebANonAttribute(Dataset):
    """
    This class builds CelebA batches that do not accomplish a given attribute
    CelebA -> http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    celeba/ folder must be inside the given path.
    Attributes must be available in list_attr_celeba.txt inside the path
    """
    def __init__(self, path='./data/', attr = 0):

        self.celeba = datasets.ImageFolder(path, transform=transforms.Compose([
                                                  transforms.Resize((64, 64)),
                                                  transforms.ToTensor(),]))

        self.labels = pd.read_csv(path + 'list_attr_celeba.txt', sep='\s+', engine='python', header=1).values
        self.labels_keys = pd.read_csv(path + 'list_attr_celeba.txt', sep='\s+', engine='python',
                                       header=1).keys().values

        mask = torch.tensor( [1 if self.labels[i][attr] == -1 else 0 for i in range(len(self.celeba))] )
        self.nsamples = mask.sum()
        self.sampler = MaskSampler(mask, self.celeba)
        self.loader = torch.utils.data.DataLoader(
                self.celeba,
                sampler=self.sampler,
                batch_size=1)
        self.iter = iter(self.loader)

    def __getitem__(self, index):
        image, label = self.iter.next()
        return image, label

    def __len__(self):
        return self.nsamples


#----------------------------------------------------------------------------------------------------------------------#

class CarsFaces(Dataset):
    """
    This class build mixed batches with images from 3D Cars dataset and 3D Faces dataset.
    3D Cars ->  http://www.cs.toronto.edu/~fidler/projects/CAD.html
    3D FACES -> https://faces.dmi.unibas.ch/bfm/index.php?nav=1-1-1&id=scans
    cars/ and faces/ folders must be inside the given path.
    """

    def __init__(self, path='./data/', split='train', p=0.5):

        self.path = path
        self.p = p
        self.cars = datasets.ImageFolder(path + 'cars/', transform=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(), ]))
        self.faces = datasets.ImageFolder(path + 'faces/', transform=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(), ]))

        if split=='train':
            self.cars = Subset(self.cars, np.arange(4000))
            self.faces = Subset(self.faces, np.arange(256))
        elif split=='test':
            self.celeba = Subset(self.cars, np.arange(4000, 4500))
            self.faces = Subset(self.faces, np.arange(256, 265))
        elif split=='val':
            self.celeba = Subset(self.cars, np.arange(4500, 4600))
            self.faces = Subset(self.faces, np.arange(265, 270))

        self.cars_loader = iter(torch.utils.data.DataLoader(self.cars, batch_size=1, shuffle=True))
        self.faces_loader = iter(torch.utils.data.DataLoader(self.faces, batch_size=1, shuffle=True))
        self.nims = int(0.9 * (self.cars.__len__() + self.faces.__len__()) )
        self.fcount_cars = 0
        self.fcount_faces = 0

    def __getitem__(self, index):

        k = int(np.round(np.random.uniform(0, 1)))
        if k <= self.p:
            k=0
        else:
            k=1
        if k == 0:
            image, label = self.cars_loader.next()
            self.fcount_cars += 1
            if self.fcount_cars == self.cars.__len__():
                self.reset_cars()
                self.fcount_cars = 0
        else:
            image, label = self.faces_loader.next()
            self.fcount_faces += 1
            if self.fcount_faces == self.faces.__len__():
                self.reset_faces()
                self.fcount_faces=0
        return image.view(image.shape[-3], image.shape[-2], image.shape[-1]), k

    def __len__(self):
        return self.nims

    def reset(self):
        self.cars_loader = iter(torch.utils.data.DataLoader(self.cars, batch_size=1, shuffle=True))
        self.faces_loader = iter(torch.utils.data.DataLoader(self.faces, batch_size=1, shuffle=True))

    def reset_faces(self):
        self.faces_loader = iter(torch.utils.data.DataLoader(self.faces, batch_size=1, shuffle=True))

    def reset_cars(self):
        self.cars_loader = iter(torch.utils.data.DataLoader(self.cars, batch_size=1, shuffle=True))


#----------------------------------------------------------------------------------------------------------------------#

class CarsFacesBatch(Dataset):
    """
    This class randomly select batches from 3D cars and 3D FACES datasets.
    For each batch, all the images come from the same dataset.
    3D Cars ->  http://www.cs.toronto.edu/~fidler/projects/CAD.html
    3D FACES -> https://faces.dmi.unibas.ch/bfm/index.php?nav=1-1-1&id=scans
    cars/ and faces/ folders must be inside the given path.
    """
    def __init__(self, path='./data/', split='train', p=0.5, batch_size=128):

        self.path = path
        self.p = p
        self.batch_size=batch_size
        self.cars = datasets.ImageFolder(path + 'cars/', transform=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(), ]))
        self.faces = datasets.ImageFolder(path + 'faces/', transform=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(), ]))

        if split=='train':
            self.cars = Subset(self.cars, np.arange(4000))
            self.faces = Subset(self.faces, np.arange(256))
        elif split=='test':
            self.celeba = Subset(self.cars, np.arange(4000, 4500))
            self.faces = Subset(self.faces, np.arange(256, 265))
        elif split=='val':
            self.celeba = Subset(self.cars, np.arange(4500, 4600))
            self.faces = Subset(self.faces, np.arange(265, 270))


        self.cars_loader = iter(torch.utils.data.DataLoader(self.cars, batch_size=1, shuffle=True))
        self.faces_loader = iter(torch.utils.data.DataLoader(self.faces, batch_size=1, shuffle=True))


        self.nims = int(0.9 * (self.cars.__len__() + self.faces.__len__()) )
        self.fcount_cars = 0
        self.fcount_faces = 0
        self.count = 0
        self.k = int(np.round(np.random.uniform(0, 1)))

    def __getitem__(self, index):

        if self.k == 0:
            image, label = self.cars_loader.next()
            self.fcount_cars += 1
            if self.fcount_cars == self.cars.__len__():
                self.reset_cars()
                self.fcount_cars = 0
        else:
            image, label = self.faces_loader.next()
            self.fcount_faces += 1
            if self.fcount_faces == self.faces.__len__():
                self.reset_faces()
                self.fcount_faces=0
        self.count += 1
        if self.count == self.batch_size:
            self.count = 0
            # self.k = int(np.round(np.random.uniform(0, 1)))
            self.k = 0 if self.k == 1 else 1


        return image.view(image.shape[-3], image.shape[-2], image.shape[-1]), self.k

    def __len__(self):
        return self.nims

    def reset(self):
        self.cars_loader = iter(torch.utils.data.DataLoader(self.cars, batch_size=1, shuffle=True))
        self.faces_loader = iter(torch.utils.data.DataLoader(self.faces, batch_size=1, shuffle=True))
    def reset_faces(self):
        self.faces_loader = iter(torch.utils.data.DataLoader(self.faces, batch_size=1, shuffle=True))

    def reset_cars(self):
        self.cars_loader = iter(torch.utils.data.DataLoader(self.cars, batch_size=1, shuffle=True))


#----------------------------------------------------------------------------------------------------------------------#

class Cars3dCars(Dataset):
    """
    This class build mixed batches with images from 3D Cars dataset and 3D Faces dataset.
    Cars dataset -> https://ai.stanford.edu/~jkrause/cars/car_dataset.html
    3D Cars ->  http://www.cs.toronto.edu/~fidler/projects/CAD.html
    cars/ and 3dcars/ folders must be inside the given path.
    """
    def __init__(self, path='./data/', p=0.5):

        self.path = path
        self.p = p
        self.cars = datasets.ImageFolder(path + 'cars/', transform=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(), ]))
        self.cars3d = datasets.ImageFolder(path + '3dcars/', transform=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(), ]))
        self.cars_loader = iter(torch.utils.data.DataLoader(self.cars, batch_size=1, shuffle=True))
        self.cars3d_loader = iter(torch.utils.data.DataLoader(self.cars3d, batch_size=1, shuffle=True))
        self.nims = int(0.9 * (self.cars.__len__() + self.cars3d.__len__()) )
        self.fcount_cars = 0
        self.fcount_cars3d = 0

    def __getitem__(self, index):

        k = int(np.round(np.random.uniform(0, 1)))
        if k <= self.p:
            k=0
        else:
            k=1
        if k == 0:
            image, label = self.cars_loader.next()
            self.fcount_cars += 1
            if self.fcount_cars == self.cars.__len__():
                self.reset_cars()
                self.fcount_cars = 0
        else:
            image, label = self.cars3d_loader.next()
            self.fcount_cars3d += 1
            if self.fcount_cars3d == self.cars3d.__len__():
                self.reset_cars3d()
                self.fcount_cars3d=0
        return image.view(image.shape[-3], image.shape[-2], image.shape[-1]), k

    def __len__(self):
        return self.nims

    def reset(self):
        self.cars_loader = iter(torch.utils.data.DataLoader(self.cars, batch_size=1, shuffle=True))
        self.cars3d_loader = iter(torch.utils.data.DataLoader(self.cars3d, batch_size=1, shuffle=True))

    def reset_cars3d(self):
        self.cars3d_loader = iter(torch.utils.data.DataLoader(self.cars3d, batch_size=1, shuffle=True))

    def reset_cars(self):
        self.cars_loader = iter(torch.utils.data.DataLoader(self.cars, batch_size=1, shuffle=True))


#----------------------------------------------------------------------------------------------------------------------#

class Cars3dCarsBatch(Dataset):
    """
    This class randomly select batches from Cars and 3D Cars datasets.
    For each batch, all the images come from the same dataset.
    Cars dataset -> https://ai.stanford.edu/~jkrause/cars/car_dataset.html
    3D Cars ->  http://www.cs.toronto.edu/~fidler/projects/CAD.html
    cars/ and 3dcars/ folders must be inside the given path.
    """
    def __init__(self, path='./data/', p=0.5, batch_size=128):

        self.path = path
        self.p = p
        self.batch_size=batch_size
        self.cars = datasets.ImageFolder(path + 'cars/', transform=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(), ]))
        self.cars3d = datasets.ImageFolder(path + '3dcars/', transform=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(), ]))

        self.cars_loader = iter(torch.utils.data.DataLoader(self.cars, batch_size=1, shuffle=True))
        self.cars3d_loader = iter(torch.utils.data.DataLoader(self.cars3d, batch_size=1, shuffle=True))
        self.nims = int(0.9 * (self.cars.__len__() + self.cars3d.__len__()) )
        self.fcount_cars = 0
        self.fcount_cars3d = 0
        self.count = 0
        self.k = int(np.round(np.random.uniform(0, 1)))

    def __getitem__(self, index):

        if self.k == 0:
            image, label = self.cars_loader.next()
            self.fcount_cars += 1
            if self.fcount_cars == self.cars.__len__():
                self.reset_cars()
                self.fcount_cars = 0
        else:
            image, label = self.cars3d_loader.next()
            self.fcount_cars3d += 1
            if self.fcount_cars3d == self.cars3d.__len__():
                self.reset_cars3d()
                self.fcount_cars3d=0
        self.count += 1
        if self.count == self.batch_size:
            self.count = 0
            # self.k = int(np.round(np.random.uniform(0, 1)))
            self.k = 0 if self.k == 1 else 1
        return image.view(image.shape[-3], image.shape[-2], image.shape[-1]), self.k

    def __len__(self):
        return self.nims

    def reset(self):
        self.cars_loader = iter(torch.utils.data.DataLoader(self.cars, batch_size=1, shuffle=True))
        self.cars3d_loader = iter(torch.utils.data.DataLoader(self.cars3d, batch_size=1, shuffle=True))

    def reset_cars3d(self):
        self.cars3d_loader = iter(torch.utils.data.DataLoader(self.cars3d, batch_size=1, shuffle=True))

    def reset_cars(self):
        self.cars_loader = iter(torch.utils.data.DataLoader(self.cars, batch_size=1, shuffle=True))


#----------------------------------------------------------------------------------------------------------------------#

class CarsChairs(Dataset):
    """
    This class build mixed batches with images from 3D Cars dataset and Chairs dataset.
    3dCars dataset -> http://www.cs.toronto.edu/~fidler/projects/CAD.html
    Chairs dataset ->  https://www.di.ens.fr/willow/research/seeing3Dchairs/
    3dcars/ and chairs/ folders must be inside the given path.
    """
    def __init__(self, path='./data/', split='train', p=0.5):

        self.path = path
        self.p = p
        self.cars = datasets.ImageFolder(path + '3dcars/', transform=transforms.Compose([
            transforms.CenterCrop(45),
            transforms.Resize((64, 64)),
            transforms.ToTensor(), ]))
        self.chairs = datasets.ImageFolder(path + 'chairs/', transform=transforms.Compose([
            transforms.CenterCrop(45),
            transforms.Resize((64, 64)),
            transforms.ToTensor(), ]))

        self.cars_loader = iter(torch.utils.data.DataLoader(self.cars, batch_size=1, shuffle=True))
        self.chairs_loader = iter(torch.utils.data.DataLoader(self.chairs, batch_size=1, shuffle=True))
        self.nims = int(0.9 * (self.cars.__len__() + self.chairs.__len__()) )
        self.fcount_cars = 0
        self.fcount_chairs = 0

    def __getitem__(self, index):

        k = int(np.round(np.random.uniform(0, 1)))
        if k <= self.p:
            k=0
        else:
            k=1
        if k == 0:
            image, label = self.cars_loader.next()
            self.fcount_cars += 1
            if self.fcount_cars == self.cars.__len__():
                self.reset_cars()
                self.fcount_cars = 0
        else:
            image, label = self.chairs_loader.next()
            self.fcount_chairs += 1
            if self.fcount_chairs == self.chairs.__len__():
                self.reset_cars()
                self.fcount_chairs=0
        return image.view(image.shape[-3], image.shape[-2], image.shape[-1]), k

    def __len__(self):
        return self.nims

    def reset(self):
        self.cars_loader = iter(torch.utils.data.DataLoader(self.cars, batch_size=1, shuffle=True))
        self.chairs_loader = iter(torch.utils.data.DataLoader(self.chairs, batch_size=1, shuffle=True))

    def reset_chairs(self):
        self.chairs_loader = iter(torch.utils.data.DataLoader(self.chairs, batch_size=1, shuffle=True))

    def reset_cars(self):
        self.cars_loader = iter(torch.utils.data.DataLoader(self.cars, batch_size=1, shuffle=True))


#----------------------------------------------------------------------------------------------------------------------#

class CarsChairsBatch(Dataset):
    """
    This class randomly select batches from 3DCars and Chairs datasets.
    For each batch, all the images come from the same dataset.
    3dCars dataset -> http://www.cs.toronto.edu/~fidler/projects/CAD.html
    Chairs dataset ->  https://www.di.ens.fr/willow/research/seeing3Dchairs/
    3dcars/ and chairs/ folders must be inside the given path.
    """
    def __init__(self, path='./data/', split='train', p=0.5, batch_size=128):

        self.path = path
        self.p = p
        self.batch_size=batch_size
        self.cars = datasets.ImageFolder(path + '3dcars/', transform=transforms.Compose([
            transforms.CenterCrop(45),
            transforms.Resize((64, 64)),
            transforms.ToTensor(), ]))
        self.chairs = datasets.ImageFolder(path + 'chairs/', transform=transforms.Compose([
            transforms.CenterCrop(45),
            transforms.Resize((64, 64)),
            transforms.ToTensor(), ]))

        self.cars_loader = iter(torch.utils.data.DataLoader(self.cars, batch_size=1, shuffle=True))
        self.chairs_loader = iter(torch.utils.data.DataLoader(self.chairs, batch_size=1, shuffle=True))


        self.nims = int(0.9 * (self.cars.__len__() + self.chairs.__len__()) )
        self.fcount_cars = 0
        self.fcount_chairs = 0
        self.count = 0
        self.k = int(np.round(np.random.uniform(0, 1)))

    def __getitem__(self, index):

        if self.k == 0:
            image, label = self.cars_loader.next()
            self.fcount_cars += 1
            if self.fcount_cars == self.cars.__len__():
                self.reset_cars()
                self.fcount_cars = 0
        else:
            image, label = self.chairs_loader.next()
            self.fcount_chairs += 1
            if self.fcount_chairs == self.chairs.__len__():
                self.reset_chairs()
                self.fcount_chairs=0
        self.count += 1
        if self.count == self.batch_size:
            self.count = 0
            # self.k = int(np.round(np.random.uniform(0, 1)))
            self.k = 0 if self.k == 1 else 1


        return image.view(image.shape[-3], image.shape[-2], image.shape[-1]), self.k

    def __len__(self):
        return self.nims

    def reset(self):
        self.cars_loader = iter(torch.utils.data.DataLoader(self.cars, batch_size=1, shuffle=True))
        self.chairs_loader = iter(torch.utils.data.DataLoader(self.chairs, batch_size=1, shuffle=True))
    def reset_chairs(self):
        self.chairs_loader = iter(torch.utils.data.DataLoader(self.chairs, batch_size=1, shuffle=True))

    def reset_cars(self):
        self.cars_loader = iter(torch.utils.data.DataLoader(self.cars, batch_size=1, shuffle=True))


#----------------------------------------------------------------------------------------------------------------------#
nchannels = {
    'celeba': 3,
    'celeba_attribute': 3,
    'celeba_nonattribute': 3,
    'light_celeba': 3,
    'mnist': 1,
    'mnist_series': 1,
    'celeba_faces': 3,
    'celeba_faces_batch': 3,
    'cars_faces': 3,
    'cars_faces_batch': 3,
    'cars_3dcars': 3,
    'cars_3dcars_batch': 3,
    'cars_chairs': 3,
    'cars_chairs_batch': 3
}


#----------------------------------------------------------------------------------------------------------------------#

def get_data(name, path='./data/', **args):
    """
    Function to obtain a dataset for the experiments in the paper
    Args:
        name: name of the dataset
        **args: possible arguments of the dataset

    Returns: torch.utils.data.Dataset objects with tr/tst/val spits of the selected dataset

    """
    # For celebA, download the dataset in ./data/celeba/
    # Put images in ./data/celeba/img/
    # Put the 'list_attr_celeba.txt' file in ./data/celeba/
    if name.lower()=='celeba':
        data_tr, data_val, data_test = CelebA(path=path + 'celeba/',
                                              transform=transforms.Compose([
                                                  transforms.Resize((64, 64)),
                                                  transforms.ToTensor(),])).get_data_tr()
    # light version of celeba for testing only with 1000 images
    elif name.lower()=='light_celeba':
        data_tr, data_val, data_test = CelebA(path=path + 'light_celeba/',
                                              transform=transforms.Compose([
                                                  transforms.Resize((64, 64)),
                                                  transforms.ToTensor(), ])).get_data_tr()

    elif name.lower()=='mnist':
        data_tr = datasets.MNIST(path, train=True, download=True,
                                 transform=transforms.ToTensor())
        data_test = datasets.MNIST(path, train=False, download=True,
                                   transform=transforms.ToTensor())
        data_val = None

    elif name.lower()=='mnist_series':
        if 'offset' in args.keys():
            offset = args['offset']
        else:
            offset = 0
        data_tr = MnistSeries(path, 'train', offset=offset)
        data_test = MnistSeries(path, 'test', offset=offset)
        data_val = None

    elif name.lower()=='celeba_faces':
        if 'p' in args.keys():
            p = args['p']
        else:
            p=0.5
        data_tr = CelebAFaces(path, 'train', p=p)
        data_test = CelebAFaces(path, 'test', p=p)
        data_val = CelebAFaces(path, 'val', p=p)

    elif name.lower()=='celeba_faces_batch':
        data_tr = CelebAFacesBatch(path, 'train')
        data_test = CelebAFacesBatch(path, 'test')
        data_val = CelebAFacesBatch(path, 'val')

    elif name.lower()=='celeba_attribute':
        data_tr = CelebAttribute(path=path + 'celeba/', attr=args['attr'])
        data_test = None
        data_val = None

    elif name.lower() == 'celeba_nonattribute':
        data_tr = CelebANonAttribute(path=path + 'celeba/', attr=args['attr'])
        data_test = None
        data_val = None

    elif name.lower() == 'cars_faces':
        data_tr = CarsFaces(path=path, split='train')
        data_test = CarsFaces(path=path, split='test')
        data_val = None

    elif name.lower() == 'cars_faces_batch':
        data_tr = CarsFacesBatch(path=path, split='train')
        data_test = CarsFacesBatch(path=path, split='test')
        data_val = None

    elif name.lower() == 'cars_3dcars':
        data_tr = Cars3dCars(path=path, split='train')
        data_test = Cars3dCars(path=path, split='test')
        data_val = None

    elif name.lower() == 'cars_3dcars_batch':
        data_tr = Cars3dCarsBatch(path=path, split='train')
        data_test = Cars3dCarsBatch(path=path, split='test')
        data_val = None

    elif name.lower() == 'cars_chairs':
        data_tr = CarsChairs(path=path, split='train')
        data_test = CarsChairs(path=path, split='test')
        data_val = None

    elif name.lower() == 'cars_chairs_batch':
        data_tr = CarsChairsBatch(path=path, split='train')
        data_test = CarsChairsBatch(path=path, split='test')
        data_val = None
        
    else:
        print('Dataset ' + name + ' not valid')
        exit()

    return data_tr, data_val, data_test