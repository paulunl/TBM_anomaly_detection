 # -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 2024

@author: unterlass/wölflingseder
"""
'''
pre-processing for VAE based anomaly detection
'''

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
# import seaborn as sns
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

tunnel = 'Synth_BBT_UT' #'Synth_BBT' # 'UT' # 'BBT'

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size_enc, hidden_size_dec, latent_size, sequence_length):
        super(VAE, self).__init__()
        
        self.hidden_size_enc = hidden_size_enc
        self.hidden_size_dec = hidden_size_dec

        self.encoder = nn.LSTM(input_size, hidden_size_enc, batch_first=True, bidirectional=True)
        
        # layers to latent size from which the reparameterization is done
        self.fc_mu = nn.LSTM(hidden_size_enc, latent_size, batch_first=True)
        self.fc_logvar = nn.LSTM(hidden_size_enc, latent_size, batch_first=True)
            
        self.decoder1 = nn.LSTM(latent_size, hidden_size_dec, batch_first=True, bidirectional=True)
        self.decoder2 = nn.LSTM(hidden_size_dec, input_size, batch_first=True)
        
        
    # sampling threw reparameterization 
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
         
        return mu + epsilon * std


    def forward(self, x):
        
        # Encoder
        out, _ = self.encoder(x)
        out = torch.add(out[:, :, :self.hidden_size_enc], torch.flip(out[:, :, self.hidden_size_enc:], [2]))/2
        out = F.leaky_relu(out)
        
        # Latent variables sampling with reparametrization trick
        mu, _ = self.fc_mu(out)
        mu = F.leaky_relu(mu)
        logvar, _ = self.fc_logvar(out)
        logvar = F.leaky_relu(logvar)
        
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
           
        dec, _ = self.decoder1(z)
        dec = torch.add(dec[:, :, :self.hidden_size_dec], torch.flip(dec[:, :, self.hidden_size_dec:], [2]))/2
        dec = F.leaky_relu(dec)
        dec, _ = self.decoder2(dec)
        
        return dec, mu, logvar
    