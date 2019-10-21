from __future__ import print_function, division
import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
from music21 import converter, instrument, note, chord, stream
# from keras.layers import Input, Dense, Reshape, Dropout, CuDNNLSTM, Bidirectional, LSTM
# from keras.layers import BatchNormalization, Activation, ZeroPadding2D
# from keras.layers.advanced_activations import LeakyReLU
# from keras.models import Sequential, Model
# from keras.optimizers import Adam
# from keras.utils import np_utils
import torch
import torch_GAN
import torch.nn as nn
import torch.optim as optim
import torch.functional as F


class Discriminator(nn.Module):
    def __init__(self, n_units):
        super(Discriminator, self).__init__()
        self.sequence_length = n_units

        self.layers = nn.Sequential(
            nn.LSTM(self.sequence_length, 512),
            nn.LSTM(512, 512, bidirectional=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        import pdb; pdb.set_trace()
        x = self.layers(x)
        return x


class Generator(nn.Module):
    def __init__(self, n_units):
        super(Generator, self).__init__()
        self.sequence_length = n_units

        self.generating_layers = nn.Sequential(
            nn.Linear(self.sequence_length, 256),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(num_features=256), # TODO: Make batchnorm work at some time
            nn.Linear(256, 1024),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(num_features=1024),
            nn.Linear(1024, self.sequence_length)
        )
        
    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = self.generating_layers(x)
        return x
        
class torchGAN():

    def __init__(self, n_units):
        self.sequence_length = n_units
        
        # create discriminator and generator 
        self.discriminator = Discriminator(self.sequence_length)
        self.generator = Generator(self.sequence_length)
    
    
    def train(self):
        pass
    
    def _generate(self):
        pass

    def _discriminate(self):
        pass
    
if __name__ == '__main__':
    n_in = 10
    ones = torch.ones(n_in, dtype=torch.float32)

    gen = Generator(n_in)
    res = gen(ones)

    disc = Discriminator(n_in)
    import pdb; pdb.set_trace()
    res = disc(res)
    import pdb; pdb.set_trace()
    print('done')
    

