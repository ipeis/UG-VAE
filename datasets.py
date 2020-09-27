

from torchvision import datasets, transforms
import pandas as pd
from torch.utils.data import *
import numpy as np
import torch
import re

class CelebA(Dataset):

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


class Mnist_Svhn(Dataset):

    def __init__(self, path='./data/', split='train'):

        self.path = path
        self.mnist= datasets.MNIST(path, train=split=='train', download=True,
                                       transform=transforms.Compose([transforms.Lambda(lambda image: image.convert('RGB')),
                                                                     transforms.Resize(32),
                                                                     transforms.ToTensor(),
                                                                     transforms.Normalize([0, 0, 0], [1, 1, 1])]))
        self.svhn = datasets.SVHN(path, split=split, download=True,
                        transform= transforms.Compose([transforms.ToTensor(),
                                                       transforms.Normalize([0, 0, 0], [1, 1, 1])]))

        self.mnist_loader = iter(torch.utils.data.DataLoader( self.mnist, batch_size=1, shuffle=True))
        self.svhn_loader = iter(torch.utils.data.DataLoader( self.svhn, batch_size=1, shuffle=True))

        self.nims = int(0.9 * (self.mnist.__len__() + self.svhn.__len__()) )

    def __getitem__(self, index):

        k = int(np.round(np.random.uniform(0, 1)))
        if k == 0:
            image, label = self.mnist_loader.next()
        else:
            image, label = self.svhn_loader.next()

        label = torch.cat([label, torch.tensor([k])])
        return image.view(3, 32, 32), label

    def __len__(self):
        return self.nims

    def reset(self):
        self.mnist_loader = iter(torch.utils.data.DataLoader(self.mnist, batch_size=1, shuffle=True))
        self.svhn_loader = iter(torch.utils.data.DataLoader(self.svhn, batch_size=1, shuffle=True))



class Mnist_Usps(Dataset):

    def __init__(self, path='./data/', split='train'):

        self.path = path
        self.mnist= datasets.MNIST(path, train=split=='train', download=True,
                                       transform= transforms.ToTensor())
        self.usps = datasets.USPS(path, train=split=='train', download=True,
                        transform=transforms.ToTensor())

        self.mnist_loader = iter(torch.utils.data.DataLoader( self.mnist, batch_size=1, shuffle=True))
        self.usps_loader = iter(torch.utils.data.DataLoader( self.usps, batch_size=1, shuffle=True))

        self.nims = int(0.9 * (self.mnist.__len__() + self.usps.__len__()) )

    def __getitem__(self, index):

        k = int(np.round(np.random.uniform(0, 1)))
        if k == 0:
            image, label = self.mnist_loader.next()
        else:
            image, label = self.usps_loader.next()

        label = torch.cat([label, torch.tensor([k])])
        print(image.shape)
        return image.view(1, 28, 28), label

    def __len__(self):
        return self.nims

    def reset(self):
        self.mnist_loader = iter(torch.utils.data.DataLoader(self.mnist, batch_size=1, shuffle=True))
        self.usps_loader = iter(torch.utils.data.DataLoader(self.usps, batch_size=1, shuffle=True))



class DigitSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, mask, data_source):
        self.mask = mask
        self.data_source = data_source

    def __iter__(self):
        return iter([i.item() for i in torch.nonzero(self.mask)])

    def __len__(self):
        return len(self.data_source)


class MnistSeries(Dataset):
    def __init__(self, path='./data/', split='train', nbatches=200, batch_size=128, offset=0, scale=1):

        if split=='train':
            self.nbatches = nbatches
        elif split=='test':
            self.nbatches = int(0.1*nbatches)
        self.batch_size = batch_size
        self.length = int(256)
        even = np.arange(self.length) * 2
        even = even*scale + offset
        even = split_digits(even)

        odd = np.arange(self.length) * 2 + 1
        odd = odd*scale + offset
        odd = split_digits(odd)

        fibonacci = [0, 1]
        for i in range(2, self.length):
            fibonacci.append(fibonacci[i - 1] + fibonacci[i - 2])
        fibonacci = np.array(fibonacci)*scale + offset
        fibonacci = split_digits(fibonacci)

        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103,
                  107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
                  227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283]
        primes = np.array(primes)*scale + offset
        primes = split_digits(primes)

        def isPrime(n):
            # see http://www.noulakaz.net/weblog/2007/03/18/a-regular-expression-to-check-for-prime-numbers/
            return re.match(r'^1?$|^(11+?)\1+$', '1' * n) == None


        N = self.length # number of primes wanted
        M = 100              # upper-bound of search space
        primes = list()           # result list

        while len(primes) < N:
            primes += filter(isPrime, range(M - 100, M)) # append prime element of [M - 100, M] to l
            M += 100                                # increment upper-bound

        #print(primes[:N]) # print result list limited to N elements

        np.array(primes) * scale + offset
        primes = split_digits(primes)


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
        """
        {'name': 'random',
         'values': np.random.randint(10, size=(self.length)) }
        """
        self.mnist = datasets.MNIST(path, train=split=='train', download=True,
                                 transform=transforms.ToTensor())
        self.loaders = []
        samplers = []
        ndigits = []
        for n in range(10):
            mask = [1 if self.mnist[i][1] == n else 0 for i in range(len(self.mnist))]
            mask = torch.tensor(mask)
            ndigits.append(mask.sum())
            samplers.append(DigitSampler(mask, self.mnist))

            self.loaders.append(torch.utils.data.DataLoader(
                self.mnist,
                sampler=samplers[-1],
                batch_size=1)
            )
        self.iters = [iter(self.loaders[n]) for n in range(len(self.loaders))]
        #self.i_s = np.zeros(len(self.series), dtype=int)
        self.i_s = np.array([np.random.randint(0, len(self.series[s]['values'])) for s in range(len(self.series))], dtype=int)
        self.s = np.random.randint(0, len(self.series))
        self.offset = offset

    def __getitem__(self, index):

        # Choose a serie
        s = torch.tensor(np.random.randint(0, len(self.series)))
        """
        if s == 4:
            # Randomize again
            self.series[4]['values'] = np.random.randint(10, size=(self.length))
        """

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
            samplers.append(DigitSampler(mask, self.mnist))

            self.loaders.append(torch.utils.data.DataLoader(
                self.mnist,
                sampler=samplers[-1],
                batch_size=1)
            )
        self.iters = [iter(self.loaders[n]) for n in range(len(self.loaders))]
        self.i_s = np.zeros(len(self.series), dtype=int)


class MnistSeries2(Dataset):
    def __init__(self, path='./data/', split='train', nbatches=200, batch_size=128, offset=0, scale=1):

        if split=='train':
            self.nbatches = nbatches
        elif split=='test':
            self.nbatches = int(0.1*nbatches)
        self.batch_size = batch_size
        self.length = int(256)

        series_2 = np.arange(self.length) * 2
        series_2 = series_2*scale + offset
        series_2 = split_digits(series_2)

        series_5 = np.arange(self.length) * 5
        series_5 = series_5 * scale + offset
        series_5 = split_digits(series_5)

        series_10 = np.arange(self.length) * 10
        series_10 = series_10 * scale + offset
        series_10 = split_digits(series_10)

        self.series = [
            {'name': 'series_2',
             'values': series_2[:self.length]},
            {'name': 'series_5',
             'values': series_5[:self.length]},
            {'name': 'series_10',
             'values': series_10[:self.length]},
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
            samplers.append(DigitSampler(mask, self.mnist))

            self.loaders.append(torch.utils.data.DataLoader(
                self.mnist,
                sampler=samplers[-1],
                batch_size=1)
            )
        self.iters = [iter(self.loaders[n]) for n in range(len(self.loaders))]
        #self.i_s = np.zeros(len(self.series), dtype=int)
        self.i_s = np.array([np.random.randint(0, len(self.series[s]['values'])) for s in range(len(self.series))], dtype=int)
        self.s = np.random.randint(0, len(self.series))
        self.offset = offset

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
            samplers.append(DigitSampler(mask, self.mnist))

            self.loaders.append(torch.utils.data.DataLoader(
                self.mnist,
                sampler=samplers[-1],
                batch_size=1)
            )
        self.iters = [iter(self.loaders[n]) for n in range(len(self.loaders))]
        self.i_s = np.zeros(len(self.series), dtype=int)


class MnistSvhnSeries(Dataset):
    def __init__(self, path='./data/', split='train', nbatches=400, batch_size=128):

        if split=='train':
            self.nbatches = nbatches
        elif split=='test':
            self.nbatches = int(0.1*nbatches)
        self.batch_size = batch_size
        self.length = int(256)
        even = np.arange(self.length) * 2
        even = split_digits(even)

        odd = np.arange(self.length) * 2 + 1
        odd = split_digits(odd)

        fibonacci = [0, 1]
        for i in range(2, self.length):
            fibonacci.append(fibonacci[i - 1] + fibonacci[i - 2])
        fibonacci = split_digits(fibonacci)
        """
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103,
                  107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
                  227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283]
        primes = split_digits(primes)

        def isPrime(n):
            # see http://www.noulakaz.net/weblog/2007/03/18/a-regular-expression-to-check-for-prime-numbers/
            return re.match(r'^1?$|^(11+?)\1+$', '1' * n) == None


        N = self.length # number of primes wanted
        M = 100              # upper-bound of search space
        primes = list()           # result list

        while len(primes) < N:
            primes += filter(isPrime, range(M - 100, M)) # append prime element of [M - 100, M] to l
            M += 100                                # increment upper-bound

        #print(primes[:N]) # print result list limited to N elements

        primes = split_digits(primes)
        """

        self.series = [
            {'name': 'even',
             'values': even[:self.length]},
            {'name': 'odd',
             'values': odd[:self.length]},
            {'name': 'fibonacci',
             'values': fibonacci[:self.length]},
            #{'name': 'primes',
            #'values': primes[self.length]},
        ]
        self.mnist = datasets.MNIST(path, train=split == 'train', download=True,
                                    transform=transforms.Compose([transforms.Lambda(lambda image: image.convert('RGB')),
                                                                  transforms.ToTensor()]))
        self.svhn = datasets.SVHN(path, split=split,
                        transform=transforms.Compose([transforms.CenterCrop(28),
                                                      transforms.ToTensor()]))
        self.mnist_loaders = []
        mnist_samplers = []
        mnist_ndigits = []
        self.svhn_loaders = []
        svhn_samplers = []
        svhn_ndigits = []
        self.mnist_masks = [torch.tensor([1 if self.mnist[i][1] == n else 0 for i in range(len(self.mnist))]) for n in range(10)]
        self.svhn_masks = [torch.tensor([1 if self.svhn[i][1] == n else 0 for i in range(len(self.svhn))]) for n in range(10)]
        for n in range(10):
            mnist_ndigits.append(self.mnist_masks[n].sum())
            mnist_samplers.append(DigitSampler(self.mnist_masks[n], self.mnist))
            svhn_ndigits.append(self.svhn_masks[n].sum())
            svhn_samplers.append(DigitSampler(self.svhn_masks[n], self.svhn))

            self.mnist_loaders.append(torch.utils.data.DataLoader(
                self.mnist,
                sampler=mnist_samplers[-1],
                batch_size=1)
            )
            self.svhn_loaders.append(torch.utils.data.DataLoader(
                self.svhn,
                sampler=svhn_samplers[-1],
                batch_size=1)
            )
        self.mnist_iters = [iter(self.mnist_loaders[n]) for n in range(len(self.mnist_loaders))]
        self.svhn_iters = [iter(self.svhn_loaders[n]) for n in range(len(self.svhn_loaders))]
        self.i_s = np.zeros(len(self.series), dtype=int)
        self.s = np.random.randint(0, len(self.series))

    def __getitem__(self, index):

        # Choose a serie
        s = torch.tensor(np.random.randint(0, len(self.series)))
        batch = []
        labels = []
        for i in range(self.batch_size):
            index = np.mod(self.i_s[s], self.length)
            n = self.series[s]['values'][index]  # digit
            self.i_s[s] += 1
            k = int(np.round(np.random.uniform(0, 1)))
            if k == 0:
                image, label = self.mnist_iters[n].next()
            else:
                image, label = self.svhn_iters[n].next()

            batch.append(image.view(3, 28, 28))
            labels.append(label)

        data = torch.cat(batch).view(-1, 3, 28, 28)
        labels = torch.cat(labels).view(-1, 1)
        labels = torch.cat((labels, torch.ones([self.batch_size, 1], dtype=torch.long)*s), dim=1).view(-1, 2)

        return data, labels

    def __len__(self):
        return self.nbatches

    def reset(self):
        self.mnist_loaders = []
        mnist_samplers = []
        mnist_ndigits = []
        self.svhn_loaders = []
        svhn_samplers = []
        svhn_ndigits = []
        for n in range(10):
            mnist_ndigits.append(self.mnist_masks[n].sum())
            mnist_samplers.append(DigitSampler(self.mnist_masks[n], self.mnist))
            svhn_ndigits.append(self.svhn_masks[n].sum())
            svhn_samplers.append(DigitSampler(self.svhn_masks[n], self.svhn))

            self.mnist_loaders.append(torch.utils.data.DataLoader(
                self.mnist,
                sampler=mnist_samplers[-1],
                batch_size=1)
            )
            self.svhn_loaders.append(torch.utils.data.DataLoader(
                self.svhn,
                sampler=svhn_samplers[-1],
                batch_size=1)
            )
        self.mnist_iters = [iter(self.mnist_loaders[n]) for n in range(len(self.mnist_loaders))]
        self.svhn_iters = [iter(self.svhn_loaders[n]) for n in range(len(self.svhn_loaders))]
        self.i_s = np.zeros(len(self.series), dtype=int)



class Faces(Dataset):

    def __init__(self, path='./data/faces/', transform=None, split='train'):

        self.path = path
        self.faces =  datasets.ImageFolder(path, transform=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(), ]))


        if split=='train':
            self.faces = Subset(self.faces, np.arange(200))
        elif split=='test':
            self.faces = Subset(self.faces, np.arange(200, 265))
        elif split=='val':
            self.faces = Subset(self.faces, np.arange(265, 270))

    def __getitem__(self, index):

        image = self.faces.__getitem__(index)

        return image

    def __len__(self):
        return self.faces.__len__()


class CelebAFaces(Dataset):

    def __init__(self, path='./data/', split='train'):

        self.path = path
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
        if k == 0:
            image, label = self.celeba_loader.next()
        else:
            image, label = self.faces_loader.next()
            self.fcount += 1
            if self.fcount == self.faces.__len__():
                self.reset_faces()
                self.fcount=0


        #label = torch.cat([label, torch.tensor(k)])
        return image.view(image.shape[-3], image.shape[-2], image.shape[-1]), k

    def __len__(self):
        return self.nims

    def reset(self):
        self.celeba_loader = iter(torch.utils.data.DataLoader(self.celeba, batch_size=1, shuffle=True))
        self.faces_loader = iter(torch.utils.data.DataLoader(self.faces, batch_size=1, shuffle=True))
    def reset_faces(self):
        self.faces_loader = iter(torch.utils.data.DataLoader(self.faces, batch_size=1, shuffle=True))


class CelebALFW(Dataset):

    def __init__(self, path='./data/', split='train'):

        self.path = path
        self.celeba = datasets.ImageFolder(path + 'celeba/', transform=transforms.Compose([
                                                  transforms.Resize((64, 64)),
                                                  transforms.ToTensor(),]))
        self.lfw = datasets.ImageFolder(path + 'lfw/', transform=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(), ]))



        celeba_tr_list = pd.read_csv(self.path+'celeba/' + 'list_eval_partition.txt', header=None).values
        celeba_partition = np.array([int(i[0][-1]) for i in celeba_tr_list])
        self.celeba_tr_idx = np.where(celeba_partition == 0)[0]
        self.celeba_val_idx = np.where(celeba_partition == 1)[0]
        self.celeba_test_idx = np.where(celeba_partition == 2)[0]

        lfw_tr_list = pd.read_csv(self.path + 'lfw/' + 'list_eval_partition.txt', header=None).values
        lfw_partition = np.array([int(i[0][-1]) for i in lfw_tr_list])
        self.lfw_tr_idx = np.where(lfw_partition == 0)[0]
        self.lfw_val_idx = np.where(lfw_partition == 1)[0]
        self.lfw_test_idx = np.where(lfw_partition == 2)[0]

        if split=='train':
            self.celeba = Subset(self.celeba, self.celeba_tr_idx)
            self.lfw = Subset(self.lfw, self.lfw_tr_idx)
        elif split=='test':
            self.celeba = Subset(self.celeba, self.celeba_test_idx)
            self.lfw = Subset(self.lfw, self.lfw_test_idx)
        elif split=='val':
            self.celeba = Subset(self.celeba, self.celeba_val_idx)
            self.lfw = Subset(self.lfw, self.lfw_val_idx)


        self.celeba_loader = iter(torch.utils.data.DataLoader(self.celeba, batch_size=1, shuffle=True))
        self.lfw_loader = iter(torch.utils.data.DataLoader(self.lfw, batch_size=1, shuffle=True))


        self.nims = int(0.9 * (self.celeba.__len__() + self.lfw.__len__()) )
        self.fcount = 0

    def __getitem__(self, index):

        k = int(np.round(np.random.uniform(0, 1)))
        if k == 0:
            image, label = self.celeba_loader.next()
        else:
            image, label = self.lfw_loader.next()
            self.fcount += 1
            if self.fcount == self.lfw.__len__():
                self.reset_lfw()
                self.fcount=0


        #label = torch.cat([label, torch.tensor(k)])
        return image.view(image.shape[-3], image.shape[-2], image.shape[-1]), k

    def __len__(self):
        return self.nims

    def reset(self):
        self.celeba_loader = iter(torch.utils.data.DataLoader(self.celeba, batch_size=1, shuffle=True))
        self.lfw_loader = iter(torch.utils.data.DataLoader(self.lfw, batch_size=1, shuffle=True))
    def reset_lfw(self):
        self.lfw_loader = iter(torch.utils.data.DataLoader(self.lfw, batch_size=1, shuffle=True))


class CelebAFacesBatch(Dataset):

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
            #self.k = int(np.round(np.random.uniform(0, 1)))
            self.k = 0 if self.k==1 else 1

        label = self.k

        #label = torch.cat([label, torch.tensor(k)])
        return image.view(image.shape[-3], image.shape[-2], image.shape[-1]), label

    def __len__(self):
        return self.nims

    def reset(self):
        self.celeba_loader = iter(torch.utils.data.DataLoader(self.celeba, batch_size=1, shuffle=True))
        self.faces_loader = iter(torch.utils.data.DataLoader(self.faces, batch_size=1, shuffle=True))
    def reset_faces(self):
        self.faces_loader = iter(torch.utils.data.DataLoader(self.faces, batch_size=1, shuffle=True))



class MnistSvhnBatch(Dataset):

    def __init__(self, path='./data/', split='train', batch_size=128):

        self.path = path
        self.mnist = datasets.MNIST(path, train=split == 'train', download=True,
                                    transform=transforms.Compose([transforms.Lambda(lambda image: image.convert('RGB')),
                                                                  transforms.Resize(32),
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize([0, 0, 0], [1, 1, 1])]))
        self.svhn = datasets.SVHN(path, split=split, download=True,
                                  transform=transforms.Compose([transforms.ToTensor(),
                                                                transforms.Normalize([0, 0, 0], [1, 1, 1])]))

        self.mnist_loader = iter(torch.utils.data.DataLoader( self.mnist, batch_size=1, shuffle=True))
        self.svhn_loader = iter(torch.utils.data.DataLoader( self.svhn, batch_size=1, shuffle=True))

        self.batch_size = batch_size

        self.nims = int(0.9 * (self.mnist.__len__() + self.svhn.__len__()) )

        self.count = 0
        self.k = int(np.round(np.random.uniform(0, 1)))
        self.fcount=0

    def __getitem__(self, index):
        if self.k == 0:
            image, label = self.mnist_loader.next()
        else:
            image, label = self.svhn_loader.next()

        self.count += 1
        if self.count == self.batch_size:
            self.count = 0
            self.k = int(np.round(np.random.uniform(0, 1)))

        label = self.k

        return image.view(3, 32, 32), label

    def __len__(self):
        return self.nims

    def reset(self):
        self.mnist_loader = iter(torch.utils.data.DataLoader(self.mnist, batch_size=1, shuffle=True))
        self.svhn_loader = iter(torch.utils.data.DataLoader(self.svhn, batch_size=1, shuffle=True))
    def reset_svhn(self):
        self.svhn_loader = iter(torch.utils.data.DataLoader(self.svhn, batch_size=1, shuffle=True))



class MnistUspsBatch(Dataset):

    def __init__(self, path='./data/', split='train', batch_size=128):

        self.path = path
        self.mnist= datasets.MNIST(path, train=split=='train', download=True,
                                       transform=transforms.ToTensor())
        self.usps = datasets.USPS(path, train=split=='train', download=True,
                                       transform=transforms.ToTensor())

        self.mnist_loader = iter(torch.utils.data.DataLoader( self.mnist, batch_size=1, shuffle=True))
        self.usps_loader = iter(torch.utils.data.DataLoader( self.usps, batch_size=1, shuffle=True))

        self.batch_size = batch_size

        self.nims = int(0.9 * (self.mnist.__len__() + self.usps.__len__()) )

        self.count = 0
        self.k = int(np.round(np.random.uniform(0, 1)))
        self.fcount=0

    def __getitem__(self, index):
        if self.k == 0:
            image, label = self.mnist_loader.next()
        else:
            image, label = self.usps_loader.next()

        self.count += 1
        if self.count == self.batch_size:
            self.count = 0
            self.k = int(np.round(np.random.uniform(0, 1)))

        label = self.k

        return image.view(image.shape[-3], image.shape[-2], image.shape[-1]), label

    def __len__(self):
        return self.nims

    def reset(self):
        self.mnist_loader = iter(torch.utils.data.DataLoader(self.mnist, batch_size=1, shuffle=True))
        self.usps_loader = iter(torch.utils.data.DataLoader(self.usps, batch_size=1, shuffle=True))
    def reset_usps(self):
        self.usps_loader = iter(torch.utils.data.DataLoader(self.usps, batch_size=1, shuffle=True))


class CelebAttribute(Dataset):

    def __init__(self, path='./data/', split='train', attr = 0):

        self.celeba = datasets.ImageFolder(path, transform=transforms.Compose([
                                                  transforms.Resize((64, 64)),
                                                  transforms.ToTensor(),]))

        self.labels = pd.read_csv(path + 'list_attr_celeba.txt', sep='\s+', engine='python', header=1).values
        self.labels_keys = pd.read_csv(path + 'list_attr_celeba.txt', sep='\s+', engine='python',
                                       header=1).keys().values

        mask = torch.tensor( [1 if self.labels[i][attr] == 1 else 0 for i in range(len(self.celeba))] )
        self.nsamples = mask.sum()
        self.sampler = DigitSampler(mask, self.celeba)
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




########################################################################################################################


nchannels = {
    'celeba': 3,
    'light_celeba': 3,
    'mnist': 1,
    'mnist_svhn': 3,
    'mnist_svhn_batch': 3,
    'mnist_usps': 1,
    'mnist_usps_batch': 3,
    'mnist_series': 1,
    'mnist_series2': 1,
    'mnist_svhn_series': 3,
    'celeba_faces': 3,
    'celeba_lfw': 3,
    'faces': 3,
    'celeba_faces_batch': 3,
    'cifar10': 3,
    'stl10': 3,
    'fashion_mnist': 1,
    'celeba_attribute': 3,
}
distributions = {
    'celeba': 'gaussian',
    'light_celeba': 'gaussian',
    'mnist': 'gaussian',
    'mnist_svhn': 'gaussian',
    'mnist_svhn_batch': 'gaussian',
    'mnist_usps': 'gaussian',
    'mnist_usps_batch': 'gaussian',
    'mnist_series': 'bernoulli',
    'mnist_series2': 'bernoulli',
    'mnist_svhn_series': 'bernoulli',
    'faces': 'gaussian',
    'celeba_faces': 'gaussian',
    'celeba_faces_batch': 'gaussian',
    'celeba_attribute': 'gaussian',
    'celeba_lfw': 'gaussian',
    'cifar10': 'gaussian',
    'stl10': 'gaussian',
    'fashion_mnist': 'bernoulli',

}


########################################################################################################################

def split_digits(int_list):

    aux = int_list.copy()
    int_list = []
    for n in aux:
        digits = list(map(int, str(n)))
        int_list+=digits
    int_list = int_list

    return int_list



########################################################################################################################

def get_data(name, **args):

    if name.lower()=='celeba':
        data_tr, data_val, data_test = CelebA(path='./data/celeba/',
                                              transform=transforms.Compose([
                                                  transforms.Resize((64, 64)),
                                                  transforms.ToTensor(),])).get_data_tr()
    elif name.lower()=='light_celeba':
        data_tr, data_val, data_test = CelebA(path='./data/light_celeba/',
                                              transform=transforms.Compose([
                                                  transforms.Resize((64, 64)),
                                                  transforms.ToTensor(), ])).get_data_tr()
    elif name.lower()=='mnist':
        data_tr = datasets.MNIST('../data', train=True, download=True,
                                 transform=transforms.ToTensor())
        data_test = datasets.MNIST('../data', train=False, download=True,
                                   transform=transforms.ToTensor())
        data_val = None

    elif name.lower()=='mnist_svhn':
        data_tr = Mnist_Svhn('./data/', 'train')
        data_test = Mnist_Svhn('./data/', 'test')
        data_val = None

    elif name.lower()=='mnist_usps':
        data_tr = Mnist_Usps('./data/', 'train')
        data_test = Mnist_Usps('./data/', 'test')
        data_val = None

    elif name.lower()=='mnist_svhn_batch':
        data_tr = MnistSvhnBatch('./data/', 'train')
        data_test = MnistSvhnBatch('./data/', 'test')
        data_val =None

    elif name.lower()=='mnist_usps_batch':
        data_tr = MnistUspsBatch('./data/', 'train')
        data_test = MnistUspsBatch('./data/', 'test')
        data_val =None


    elif name.lower()=='mnist_series':
        if 'offset' in args.keys():
            offset = args['offset']
        else:
            offset = 0
        data_tr = MnistSeries('./data/', 'train', offset=offset)
        data_test = MnistSeries('./data/', 'test', offset=offset)
        data_val = None
    elif name.lower()=='mnist_series2':
        if 'offset' in args.keys():
            offset = args['offset']
        else:
            offset = 0
        data_tr = MnistSeries2('./data/', 'train', offset=offset)
        data_test = MnistSeries2('./data/', 'test', offset=offset)
        data_val = None

    elif name.lower()=='mnist_svhn_series':
        data_tr = MnistSvhnSeries('./data/', 'train')
        data_test = MnistSvhnSeries('./data/', 'test')
        data_val = None

    elif name.lower()=='faces':
        data_tr = Faces('./data/faces/', 'train')
        data_test = Faces('./data/faces/', 'test')
        data_val = Faces('./data/faces/', 'val')

    elif name.lower()=='celeba_faces':
        data_tr = CelebAFaces('./data/', 'train')
        data_test = CelebAFaces('./data/', 'test')
        data_val = CelebAFaces('./data/', 'val')

    elif name.lower()=='celeba_faces_batch':
        data_tr = CelebAFacesBatch('./data/', 'train')
        data_test = CelebAFacesBatch('./data/', 'test')
        data_val = CelebAFacesBatch('./data/', 'val')

    elif name.lower()=='celeba_lfw':
        data_tr = CelebALFW('./data/', 'train')
        data_test = CelebALFW('./data/', 'test')
        data_val = CelebALFW('./data/', 'val')

    elif name.lower()=='cifar10':
        data_tr = datasets.CIFAR10('../data', train=True, download=True,
                                 transform=transforms.Compose([
                                                  transforms.Resize((64, 64)),
                                                  transforms.ToTensor()]))
        data_test = datasets.CIFAR10('../data', train=False, download=True,
                                   transform=transforms.Compose([
                                                  transforms.Resize((64, 64)),
                                                  transforms.ToTensor()]))
        data_val = None

    elif name.lower()=='stl10':
        data_tr = datasets.STL10('../data', split='train', download=True,
                                 transform=transforms.Compose([
                                                  transforms.Resize((64, 64)),
                                                  transforms.ToTensor()]))
        data_test = datasets.STL10('../data', split='test', download=True,
                                   transform=transforms.Compose([
                                                  transforms.Resize((64, 64)),
                                                  transforms.ToTensor()]))
        data_val = None

    elif name.lower()=='fashion_mnist':
        data_tr = datasets.FashionMNIST('../data', train=True, download=True,
                                        transform=transforms.ToTensor())
        data_test = datasets.FashionMNIST('../data', train=True, download=True,
                                   transform=transforms.ToTensor())
        data_val = None

    elif name.lower()=='celeba_attribute':
        data_tr = CelebAttribute(path='./data/celeba/', attr=args['attr'], split='train')
        data_test = None
        data_val = None

    else:
        print('Dataset not valid')
        exit()

    return data_tr, data_val, data_test