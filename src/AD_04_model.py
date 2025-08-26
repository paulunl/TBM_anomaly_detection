# -*- coding: TBM_Bf-8 -*-
"""
    TBM Operational Data-Driven Anomaly Detection in Hard Rock Excavations

    ---- link to paper
    DOI: XXXX

    Script containing the code of the VAE model.

    @author: Paul Unterlaß / Mario Wölflingseder
"""

# =============================================================================
# Imports
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# VAE Model Architecture
# =============================================================================
class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) Model for anomaly detection.

    Parameters:
    - input_size: Size of the input feature vector
    - hidden_size_enc: Hidden size for the encoder LSTM
    - hidden_size_dec: Hidden size for the decoder LSTM
    - latent_size: Size of the latent (bottleneck) layer
    - sequence_length: Length of input sequences
    
    Attributes:
    - encoder: LSTM layer for encoding input sequences
    - fc_mu: LSTM layer for the mean of the latent space
    - fc_logvar: LSTM layer for the log variance of the latent space
    - decoder1: LSTM layer for decoding the latent representation back to hidden state
    - decoder2: LSTM layer for reconstructing the input from the decoder's output
    """
    
    def __init__(self, input_size, hidden_size_enc, hidden_size_dec, latent_size, sequence_length):
        """
        Initialize the layers of the VAE model.
        """
        super(VAE, self).__init__()

        # Encoder LSTM layer (Bidirectional)
        self.encoder = nn.LSTM(input_size, hidden_size_enc, batch_first=True, bidirectional=True)
        
        # Layers to produce the latent variables (mean and log variance)
        self.fc_mu = nn.LSTM(hidden_size_enc, latent_size, batch_first=True)
        self.fc_logvar = nn.LSTM(hidden_size_enc, latent_size, batch_first=True)
            
        # Decoder LSTM layers (Bidirectional for the first decoder)
        self.decoder1 = nn.LSTM(latent_size, hidden_size_dec, batch_first=True, bidirectional=True)
        self.decoder2 = nn.LSTM(hidden_size_dec, input_size, batch_first=True)
        
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for the VAE model:
        This generates latent variable `z` from the mean (`mu`) and log-variance (`logvar`).
        
        Parameters:
        - mu: The mean of the latent distribution
        - logvar: The log variance of the latent distribution
        
        Returns:
        - z: The reparameterized latent variable
        """
        std = torch.exp(0.5 * logvar)  # Standard deviation from log variance
        epsilon = torch.randn_like(std)  # Sample epsilon from normal distribution
        
        # Return latent variable using reparameterization trick
        return mu + epsilon * std

    def forward(self, x):
        """
        Forward pass through the VAE model.
        
        Parameters:
        - x: Input tensor
        
        Returns:
        - dec: Reconstructed input sequence
        - mu: Latent mean
        - logvar: Latent log-variance
        """
        
        # Encoder
        out, _ = self.encoder(x)
        
        # Concatenate the forward and backward LSTM outputs and apply mean
        out = torch.add(out[:, :, :self.hidden_size_enc], torch.flip(out[:, :, self.hidden_size_enc:], [2])) / 2
        out = F.leaky_relu(out)  # Apply LeakyReLU activation function
        
        # Latent variable sampling using the reparameterization trick
        mu, _ = self.fc_mu(out)
        mu = F.leaky_relu(mu)  # Apply LeakyReLU activation to the mean
        logvar, _ = self.fc_logvar(out)
        logvar = F.leaky_relu(logvar)  # Apply LeakyReLU activation to the log variance
        
        # Reparameterization: Sample z from the latent space
        if self.training:
            z = self.reparameterize(mu, logvar)  # During training, sample z
        else:
            z = mu  # During inference, use the mean as the latent variable
            
        # Decoder part 1 (Bidirectional LSTM)
        dec, _ = self.decoder1(z)
        dec = torch.add(dec[:, :, :self.hidden_size_dec], torch.flip(dec[:, :, self.hidden_size_dec:], [2])) / 2
        dec = F.leaky_relu(dec)  # Apply LeakyReLU activation
        
        # Decoder part 2 (Final LSTM to reconstruct the input sequence)
        dec, _ = self.decoder2(dec)
        
        # Return reconstructed sequence, latent mean, and log variance
        return dec, mu, logvar
