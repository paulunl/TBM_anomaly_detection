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
    

def train_vae(vae,
              train_dataloader,
              test_dataloader1,
              test_dataloader2,
              test_dataloader3,
              val_dataloader,
              num_epochs,
              learning_rate,
              beta,
              file_name
              ):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device)

    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction='sum').to(device)
    
    best_eval_loss = float("Inf")

    # history for plotting learning curve
    history = dict(train_rec=[], train_kl=[], 
                   test_rec=[], test_kl=[],
                   val_rec=[], val_kl=[],
                   )
    
    for epoch in range(num_epochs):
        
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        
        # storing losses of epoch
        losses = dict(train_rec = [], train_kl = [], 
                      test_rec = [], test_kl = [],
                      val_rec = [], val_kl = [])
        
        vae = vae.train()
        
        # training loop over iterations
        for input_data, _ in tqdm(train_dataloader):
            
            input_data = input_data.to(device).float()
            reconstructed_data, mu, logvar = vae(input_data)

            # Compute loss for backprop
            reconstruction_loss = criterion(reconstructed_data, input_data)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = reconstruction_loss + beta * kl_div
            
            losses['train_rec'].append(reconstruction_loss.item())
            losses['train_kl'].append(beta*kl_div.item())

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # computes validation losses for learning curve
        vae = vae.eval()
        
        with torch.no_grad():

            for input_data, _ in tqdm(test_dataloader1):
                
                input_data = input_data.to(device).float()
                reconstructed_data, mu, logvar = vae(input_data)
                # Compute validation loss
                reconstruction_loss = criterion(reconstructed_data, input_data)
                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = reconstruction_loss + beta * kl_div
                
                losses['test_rec'].append(reconstruction_loss.item())
                losses['test_kl'].append(beta*kl_div.item())
            
            for input_data, _ in tqdm(test_dataloader2):
    
                input_data = input_data.to(device).float()
                reconstructed_data, mu, logvar = vae(input_data)
                # Compute validation loss
                reconstruction_loss = criterion(reconstructed_data, input_data)
                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = reconstruction_loss + beta * kl_div
                
                losses['test_rec'].append(reconstruction_loss.item())
                losses['test_kl'].append(beta*kl_div.item())
    
            for input_data, _ in tqdm(test_dataloader3):
    
                input_data = input_data.to(device).float()
                reconstructed_data, mu, logvar = vae(input_data)
                # Compute validation loss
                reconstruction_loss = criterion(reconstructed_data, input_data)
                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = reconstruction_loss + beta * kl_div
                
            for input_data, _ in tqdm(val_dataloader):
                
                input_data = input_data.to(device).float()
                reconstructed_data, mu, logvar = vae(input_data)
                # Compute validation loss
                reconstruction_loss = criterion(reconstructed_data, input_data)
                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = reconstruction_loss + beta * kl_div
                
                losses['val_rec'].append(reconstruction_loss.item())
                losses['val_kl'].append(beta*kl_div.item())
        
        # computing mean values of losses of epoch, saving it to history and printing it
        for key, value  in losses.items():
            
            epoch_loss = np.mean(losses[key])
            history[key].append(epoch_loss)
            print(f'{key}: {epoch_loss:.8f}')
        
        # saving model at current state and history of losses
        eval_loss = np.mean(losses['val_rec']) +  np.mean(losses['val_kl'])
        
        if best_eval_loss > eval_loss:
            best_eval_loss = eval_loss
        
            torch.save(vae,
                       fr'02_Results\{tunnel}\01_Models\{file_name}' + '.pth')
            torch.save(history,
                       fr'02_Results\{tunnel}\01_Models\history of ' + f'{file_name}' + '.pt')

    print(f'TRAINING COMPLETE: {file_name}' + '.pth')
    
    # creats plot of learning curve with plt
    ax = plt.figure().gca()
    ax.plot(history['train_rec'])
    ax.plot(history['test_rec'])
    ax.plot(history['val_rec'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xlim(0, num_epochs)
    plt.legend(['train', 'test', 'validation'])
    plt.title('Loss over training epochs')
    plt.savefig(fr'02_Results\{tunnel}\02_Plots\01_learning_curve\Learning Curve_{file_name}.png')
    plt.show()
    
    return vae.eval(), history


# observing reconstruction errors for anomaly detection to help find a threshhold
def test_vae(vae,
             dataloader
             ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device)

    error_list = []

    vae.eval()
    with torch.no_grad():
        for input_data, _ in dataloader:
            
            # initial_values = input_data[:, 0, :]
            # input_data -= torch.roll(input_data, 1, 1)
            # input_data[:, 0, :] = initial_values
            
            input_data = input_data.to(device).float()

            # Forward pass
            reconstructed_data, mu, logvar = vae(input_data)
            
            # reconstruction error for anomaly detection
            error = torch.mean((reconstructed_data - input_data) ** 2, dim=(1, 2)).cpu()
            error = error.numpy().flatten()
            for i in error:
                error_list.append(i) 
    
    return error_list, mu, logvar


# plots reconstruction error and threshold and GI as backgroundcolor
def plot_reconstruction_error(error_list,
                              dataloader,
                              sequence_length,
                              start_km,
                              title,
                              file_name,
                              threshold=1000,
                              ):
    
    # plots distribution of amounts of certain reconstruction errors
    # sns.displot(error_list, bins=50, kde=True);

    # makes a list of the labels from the input data
    label_list = []
    for _, label in dataloader:
        for label in label:
            label = label.numpy()
            label = float(label)
            label_list.append(label)
            
    # making list of thresholds in length of amount of errors
    threshold_for_plot = []
    for i in range(len(error_list)):
        threshold_for_plot.append(threshold)
        
    # creates dataframe with the lists of the three needed values    
    df = pd.DataFrame(list(zip(
                    error_list,
                    threshold_for_plot,
                    label_list)), columns = ['Error', 'Threshold', 'Label'])
    
    # # arrange list for Tunnel Distance at x axis
    df['Tunnel Distance [km]'] = np.arange(start_km, (start_km*1000 + 0.05*(len(error_list)-0.9))/1000, 0.05/1000) # -0.9 only because the length somehow doesnt match without it
    
    # shifting Error for half of sequence length to have it compared in the middle of SL
    df['Error'] = df['Error'].shift(int(sequence_length/2))
    
    # cutting out first part of section (half of seq length)
    df = df[int(sequence_length/2):]
    df = df.reset_index()
    
    # plotting reconstruction errors
    fig, ax = plt.subplots(1,1, figsize=(15,10))
    
    ax.scatter(df['Tunnel Distance [km]'], df['Error'], s=2, c='black')
    plt.xlim(start_km + 0.05*int(sequence_length/2)/1000,
             start_km + 0.05*len(error_list)/1000)
    plt.ylim(0, df['Error'].max())
    
    # plotting threshhold if one is given to the function
    if threshold != 1000 : ax.plot(df['Threshold'], color="black",
                                           linewidth=2, linestyle='--') 

    plt.ylabel('Reconstruction Error', fontsize='16')
    plt.xlabel('Tunnel Distance [km]', fontsize='16')
    plt.title(f'{title}', fontsize='16')
    ax.xaxis.set_ticks(np.arange(
        start_km, (start_km*1000 + 0.05*(len(error_list)-0.9))/1000, 0.1))
    plt.xticks(rotation=90, fontsize='16')
    plt.yticks(fontsize='16')
        
    n_clusters = df['Label'].max() - df['Label'].min() + 1
    
    # cmap = plt.get_cmap('RdYlGn_r', n_clusters) # autumn_r
    if n_clusters == 6:
        cmap = (mpl.colors.ListedColormap(['green',
                                       'greenyellow',
                                       'gold',
                                       'orange',
                                       'darkorange',
                                       'red'])
                .with_extremes(over='red', under='green'))
        
    elif n_clusters == 5:
        cmap = (mpl.colors.ListedColormap(['green',
                                       'greenyellow',
                                       'gold',
                                       'orange',
                                       'darkorange'])
                .with_extremes(over='red', under='green'))
    elif n_clusters == 4 and tunnel == 'Synth_BBT':
        cmap = (mpl.colors.ListedColormap([
                                       'green',
                                       'gold',
                                       'orange',
                                       'red'])
                .with_extremes(over='red', under='green'))
            
    elif n_clusters == 4 and df['Label'].min() == 0:
        cmap = (mpl.colors.ListedColormap(['green',
                                       'greenyellow',
                                       'gold',
                                       'orange'])
                .with_extremes(over='red', under='green'))
    
    elif n_clusters == 4 and df['Label'].min() == 1:
        cmap = (mpl.colors.ListedColormap([
                                       'greenyellow',
                                       'gold',
                                       'orange',
                                       'darkorange'])
                .with_extremes(over='red', under='green'))
        
    elif n_clusters == 3:
        cmap = (mpl.colors.ListedColormap(['green',
                                       'gold',
                                       'red'])
                .with_extremes(over='red', under='green'))
        
    elif n_clusters == 2:
        cmap = (mpl.colors.ListedColormap(['green',
                                       'gold'])
                .with_extremes(over='red', under='green'))
        
    else:
        print('colour map not suited for the data to be plotted')


    c = ax.pcolorfast(ax.get_xlim(),
                      ax.get_ylim(),
                      df['Label'].values[np.newaxis],
                      cmap=cmap,
                      vmin=(df['Label'].min()),
                      vmax=(df['Label'].max()+1),
                      alpha=0.5)
    
        
    # bounds = [0, 1, 2, 3, 4, 5]
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)    
    # cbar = fig.colorbar(
    #         mpl.cm.ScalarMappable(cmap=cmap, norm=norm), c,
    #         cax=ax, orientation='vertical',
    #         extend='both', extendfrac='auto',
    #         spacing='uniform',
    #         # label='Custom extension lengths, some other units',
    #     )

    cbar = fig.colorbar(c, ax=ax) # , spacing='uniform'
    
    tick_locs = np.linspace(int(df['Label'].min()),
                            int(df['Label'].max()+1),
                            int(2 * n_clusters + 1))[1::2]
    
    tick_label = np.arange(int(df['Label'].min()),
                                int(df['Label'].max() + 1))
    
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(tick_label)
    
    cbar.ax.set_ylim(df['Label'].min(), int(df['Label'].max()+1))
    
    ax.legend(['Reconstruction Error', 'Threshold'],
              fontsize='12',
              loc='upper left')

    plt.tight_layout()
    plt.savefig(fr'02_Results\{tunnel}\02_Plots\02_errors_vs_classes\{title}_{file_name}.png')

    return df

def plot_original_data(df, start_km, end_km, feature, title):
    
    # creating section for test dataset
    df_section = df[(df['Tunnel Distance [m]'] >= start_km*1000) & 
                      (df['Tunnel Distance [m]'] < end_km*1000)]   
    df_section = df_section.reset_index()
    
    # plotting
    orig_plot = plt.figure().gca()
    
    orig_plot.plot(df_section['Tunnel Distance [m]'],
                   df_section[feature],
                   linestyle='',
                   marker='o',
                   markersize=1)
    
    # orig_plot.plot(df['Error'], linestyle='', marker='o', markersize=1)
    
    plt.xlim(start_km*1000, end_km*1000)
    
    # plotting GI as  different background colors, alternativ colors: RdYlGn_r; Accent
    orig_plot.pcolorfast(orig_plot.get_xlim(),
                         orig_plot.get_ylim(),
                         df_section['Class'].values[np.newaxis],
                         cmap='RdYlGn_r',
                         alpha=0.2)
    
    plt.ylabel(feature)
    plt.xlabel('TUNNEL DISTANCE [m]')
    plt.title(f'{title}', fontsize='10')
    orig_plot.xaxis.set_ticks(np.arange(start_km*1000, end_km*1000, 100))
    plt.xticks(rotation=90)
    
    plt.show()
    
    return df_section

# plots examples for reconstructed sequences
def plot_reconstructed_seq(vae, dataloader, km=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device)
    
    reconstructed_data_list = []
    input_data_list =[]

    vae.eval()
    with torch.no_grad():
        for input_data, _ in dataloader:
            
            # initial_values = input_data[:, 0, :]
            # input_data -= torch.roll(input_data, 1, 1)
            # input_data[:, 0, :] = initial_values
            
            input_data = input_data.to(device).float()

            # Forward pass
            reconstructed_data, _, _ = vae(input_data)
        
            reconstructed_data_np = reconstructed_data.cpu().numpy()
            for i in reconstructed_data_np:
                reconstructed_data_list.append(i)
                
            input_data_np = input_data.cpu().numpy()
            for i in input_data_np:
                input_data_list.append(i) 
    
    # sns.lineplot(data=reconstructed_data.flatten(2).detach().numpy().squeeze())
    # sns.lineplot(data=input_data.flatten(2).detach().numpy().squeeze())
    plt.show()
    
    # creats plot with plt
    ax = plt.figure().gca()
    ax.plot(input_data_list[int(km*1000*20)], 
            label=('spec. penetration - input','torque ratio - input'),
            color='red'
            )
    ax.plot(reconstructed_data_list[int(km*1000*20)], 
            label=('spec. penetration - reconstr','torque ratio - reconstr'),
            color='green'
            )
    plt.ylabel('Scaled feature values')
    plt.xlabel('Vectors of sequence')
    plt.title(f'Sequence starting at {km} km', fontsize='10')
    plt.legend(fontsize="8", loc='upper right')
    plt.tight_layout()
    # plt.savefig(f'Sequence_starting_at_{km}km.png')
    plt.show()
    
    return reconstructed_data_list, input_data_list

            
# detects how many sequences are exceeding the reconstruction error threshold
def detect_anomalies(vae, dataloader, threshold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device)

    vae.eval()
    with torch.no_grad():
        anomaly_count = 0
        no_anomaly_count = 0
        
        for input_data, _ in dataloader:
            
            # initial_values = input_data[:, 0, :]
            # input_data -= torch.roll(input_data, 1, 1)
            # input_data[:, 0, :] = initial_values
            
            input_data = input_data.to(device).float()
            
            # Forward pass
            reconstructed_data, _, _ = vae(input_data)

            # Calculate reconstruction error (MSE loss)
            error = torch.mean((reconstructed_data - input_data) ** 2,
                               dim=(1, 2))
            
            # Detect anomalies based on the threshold
            anomalies = error > threshold
            no_anomalies = error <= threshold
            
            # counts anomaly sequences and non_anomaly sequences
            anomaly_count += torch.sum(anomalies).item()
            no_anomaly_count += torch.sum(no_anomalies).item()

        print(f"Total anomalies detected:{anomaly_count}/{anomaly_count + no_anomaly_count}")
        
        return anomaly_count, no_anomaly_count
