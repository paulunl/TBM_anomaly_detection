 # -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 2024

@author: unterlass/wölflingseder
"""
'''
Code for the training, testing, of the model and the evaluation of the anomaly
detection

'''

import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.stats.stattools
import seaborn as sns
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter

from VAE_05_model import VAE
from VAE_02_preprocessor_for_VAE import TBMDataset

# # =============================================================================
# # Function that creates torch datasets
# # =============================================================================

# class TBMDataset(Dataset):
    
#     def __init__(self, sequences):
#         self.sequences = sequences

#     def __len__(self):
#         return len(self.sequences)
    
#     def __getitem__(self, idx):
#         sequence, label = self.sequences[idx]
#         sequence = torch.tensor(sequence.to_numpy())
#         # if sequence.shape[1] != 8 or sequence.shape[1] != 5:
#         #     raise RuntimeError(f"Expected input size 8 or 5, got {sequence.shape}")
#         label = torch.tensor(label)
        
#         return sequence, label

# =============================================================================
# Hyperparameters
# =============================================================================
tunnel = 'BBT' #'Synth_BBT' #'UT'

if tunnel == 'Synth_BBT' or tunnel == 'Synth_BBT_UT':
    input_size = 5  # Size of input features
else:
    input_size = 8

hidden_size_encoder = 30  # Size of hidden state in LSTM layers of encoder
hidden_size_decoder = 30 # Size of hidden state in LSTM layers of decoder
latent_size = 3  # Size of the latent representation
batch_size = 64
num_epochs =  50
learning_rate = 0.0002
sequence_length = 100
beta = 0.001

# =============================================================================
# Load datasets and create DataLoader
# =============================================================================
if tunnel == 'UT':
    CLASS = 4
    start_val = 3.5
    end_val = 4.5
    start_test_1 = 2.4
    end_test_1 = 2.8
    start_test_2 = 4.9
    end_test_2 = 5.25
    start_test_3 = 5.9
    end_test_3 = 6.2
    
elif tunnel == 'Synth_BBT_UT':
    CLASS = 4
    start_val = 0
    end_val = 0.25
    start_test_1 = 2.4
    end_test_1 = 2.8
    start_test_2 = 4.9
    end_test_2 = 5.25
    start_test_3 = 5.9
    end_test_3 = 6.2
    
elif tunnel == 'Synth_BBT':
    CLASS = 3
    start_val = 0
    end_val = 0.25
    start_test_1 = 3.5
    end_test_1 = 4.5
    start_test_2 = 5.0
    end_test_2 = 6.0
    start_test_3 = 9.5
    end_test_3 = 10.5
    
elif tunnel == 'BBT':
    CLASS = 3
    start_val = 10.5
    end_val = 11.5
    start_test_1 = 3.5
    end_test_1 = 4.5
    start_test_2 = 5.0
    end_test_2 = 6.0
    start_test_3 = 9.5
    end_test_3 = 10.5

else:
    print('tunnel not defined')
    
file_name = f'{tunnel}_CLASS{CLASS}_seq{sequence_length}_beta{beta}_ep{num_epochs}_hse{hidden_size_encoder}_hsd{hidden_size_decoder}_ls{latent_size}'

# train dataset
train_dataset = torch.load(
    f'01_data/{tunnel}/datasets/{tunnel}_{CLASS}_train_dataset_seq{sequence_length}.pt'
    )

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False
    )

# test 1-3 datasets
test_dataset1 = torch.load(
    f'01_data/{tunnel}/datasets/{tunnel}_{CLASS}_test_dataset1_seq{sequence_length}_{start_test_1}-{end_test_1}.pt')

test_dataloader1 = DataLoader(
    test_dataset1,
    batch_size=batch_size,
    shuffle=False
    )

test_dataset2 = torch.load(
    f'01_data/{tunnel}/datasets/{tunnel}_{CLASS}_test_dataset2_seq{sequence_length}_{start_test_2}-{end_test_2}.pt')

test_dataloader2 = DataLoader(
    test_dataset2,
    batch_size=batch_size,
    shuffle=False
    )

test_dataset3 = torch.load(
    f'01_data/{tunnel}/datasets/{tunnel}_{CLASS}_test_dataset3_seq{sequence_length}_{start_test_3}-{end_test_3}.pt')

test_dataloader3 = DataLoader(
    test_dataset3,
    batch_size=batch_size,
    shuffle=False)

# validation dataset
val_dataset = torch.load(
    f'01_data/{tunnel}/datasets/{tunnel}_{CLASS}_val_dataset_seq{sequence_length}_{start_val}-{end_val}.pt'
    )

val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
    )


# =============================================================================
# model
# =============================================================================
# create model
vae = VAE(input_size,
          hidden_size_encoder,
          hidden_size_decoder,
          latent_size,
          sequence_length
          )

# # load model
# model = torch.load(f'02_Results/{tunnel}/01_Models/{file_name}' + '.pth')
# history = torch.load(f'02_Results/{tunnel}/01_Models/history of {file_name}.pt')


# =============================================================================
# training
# =============================================================================
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


# Train the VAE / plot learning curve 
model, history = train_vae(vae,
                           train_dataloader,
                           test_dataloader1,
                           test_dataloader2,
                           test_dataloader3,
                           val_dataloader,
                           num_epochs,
                           learning_rate,
                           beta,
                           file_name
                           )

# # load best model after training model
# model = torch.load(f'02_Results/UT/01_Models/{file_name}' + '.pth')
# history = torch.load(f'02_Results/UT/01_Models/history of {file_name}.pt')

# =============================================================================
# test the model
# =============================================================================
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

error_test1, lat_mu_test1, lat_logvar_test1 = test_vae(model,
                                                       test_dataloader1,
                                                       )
error_test2, lat_mu_test2, lat_logvar_test2 = test_vae(model,
                                                       test_dataloader2,
                                                       )
error_test3, lat_mu_test3, lat_logvar_test3 = test_vae(model,
                                                       test_dataloader3,
                                                       )
error_test_sum = error_test1 + error_test2 + error_test3

# =============================================================================
# threshold
# =============================================================================

# set threshold for every test section and plot histogram
def threshold(error_test, title):
    error_test = np.array(error_test)
    print('highest error value', error_test.max())
    
    quantile = np.quantile(error_test, 0.99)
    print('99% quantil:', quantile)

    q1 = np.percentile(error_test, 25)
    q3 = np.percentile(error_test, 75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    print('upper boundary:', upper_bound)
    
    # calculate the medcouple
    mc = statsmodels.stats.stattools.medcouple(error_test, axis=0)
    print('medcouple:', mc)
    
    # negative medcouple signifies a negative skewed distribution  resulting in
    # different upper and lower bounds
    if mc < 0:
        adjusted_lower_bound = q1 - 1.5*np.exp(-3*mc)*iqr
        adjusted_upper_bound = q3 + 1.5*np.exp(4*mc)*iqr
    else:
        adjusted_lower_bound = q1 - 1.5*np.exp(-4*mc)*iqr
        adjusted_upper_bound = q3 + 1.5*np.exp(3*mc)*iqr
    
    print('skewness adjusted lower boundary:', adjusted_lower_bound)
    print('skewness adjusted upper boundary:', adjusted_upper_bound)
        
    # plot error
    sns.set(style="ticks")

    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, figsize=(15, 10),
                                        gridspec_kw={"height_ratios": (.15, .85)})
    
    # set right boxplot whisker for plotting
    if mc < 0:
        whis = q3 + 1.5*np.exp(4*mc)
    else:
        whis = q3 + 1.5*np.exp(3*mc)

    sns.boxplot(error_test,
                ax=ax_box,
                orient='h',
                whis=whis,
                color='lightgrey',
                linewidth=3)
    
    sns.distplot(error_test, ax=ax_hist) # .set_title(f'{title}')
    
    # if error_test.max() > 0.1:
    #     x_max = error_test.max() + 0.06
    # elif skewed_boundary_up > error_test.max():
    #     x_max = skewed_boundary_up + 0.005
    # else:
    #     x_max = error_test.max()

    if error_test.max() > 0.1:
        x_max = adjusted_upper_bound + 0.1
        
    elif adjusted_upper_bound < 0.5:
        x_max = adjusted_upper_bound + 0.005
        
    else:
        x_max = error_test.max()
            
    plt.xlim(xmin=0, xmax = x_max)
    plt.ylabel('Instances', fontsize='16')
    plt.xlabel('Reconstruction Error [-]', fontsize='16')
    plt.xticks(rotation=90, fontsize='16')
    plt.yticks(fontsize='16')
    plt.tight_layout()
    
    plt.subplots_adjust(top=0.95)
    plt.suptitle(f'{title}', fontsize = 16)
    plt.axvline(upper_bound, c='black', ls='--', label='upper boxplot boundary')
    plt.axvline(adjusted_upper_bound, c='red', ls='--', label='skewness adjusted \nupper boxplot boundary')
    ax_box.set(yticks=[])
    sns.despine(ax=ax_hist)
    sns.despine(ax=ax_box, left=True)
    ax_hist.legend(loc='upper right', facecolor='white',
                   edgecolor='black', framealpha=1, prop={'size': 16})
    plt.savefig(fr'02_Results\{tunnel}\02_Plots\03_threshold\{title}_{file_name}.png', dpi=300)

    return adjusted_upper_bound, upper_bound

adjusted_threshold_test1, threshold_test1 = threshold(error_test1, 'test_set_1')
adjusted_threshold_test2, threshold_test2 = threshold(error_test2, 'test_set_2')
adjusted_threshold_test3, threshold_test3 = threshold(error_test3, 'test_set_3')
adjusted_threshold_all, threshold_all = threshold(error_test_sum, 'all_test_sets')

# =============================================================================
# plot reconstruction errors
# =============================================================================
def plot_reconstruction_error(error_list,
                              dataloader,
                              sequence_length,
                              start_km,
                              title,
                              file_name,
                              threshold,
                              adjusted_threshold
                              ):
    

    # creates a list of the labels from the input data
    label_list = []
    for _, label in dataloader:
        for label in label:
            label = label.numpy()
            label = float(label)
            label_list.append(label)
            
    # create list of thresholds in length of amount of errors
    threshold_for_plot = []
    adjusted_threshold_for_plot = []
    for i in range(len(error_list)):
        threshold_for_plot.append(threshold)
        adjusted_threshold_for_plot.append(adjusted_threshold)
    
    # create dataframe with the lists of the three needed values    
    df = pd.DataFrame(list(zip(
                    error_list,
                    threshold_for_plot,
                    adjusted_threshold_for_plot,
                    label_list)), columns = ['Error', 'Threshold',
                                             'adjusted Threshold', 'Label'])
    
    # arrange list for Tunnel Distance at x axis
    df['Tunnel Distance [km]'] = np.arange(start_km, (start_km*1000 + 0.05*(len(error_list)-0.9))/1000, 0.05/1000) # -0.9 only because the length somehow doesnt match without it
    
    # shifting Error for half of sequence length to have it compared in the middle of SL
    df['Error'] = df['Error'].shift(int(sequence_length/2))
    
    # cutting out first part of section (half of seq length)
    df = df[int(sequence_length/2):]
    df = df.reset_index()
    
    # plotting reconstruction errors
    fig, ax = plt.subplots(1,1, figsize=(10,6))
    
    ax.scatter(df['Tunnel Distance [km]'], df['Error'], s=2, c='black')
    plt.xlim(start_km + 0.05*int(sequence_length/2)/1000,
             start_km + 0.05*len(error_list)/1000)
    plt.ylim(0, df['Error'].max()+0.012)
    
    # plotting threshhold and skewness adjusted threshold
    ax.plot(df['Threshold'], color="black", linewidth=2, linestyle='--') 
    ax.plot(df['adjusted Threshold'], color='red', linewidth=2, linestyle='--')

    plt.ylabel('Reconstruction Error', fontsize='16')
    plt.xlabel('Tunnel Distance [m]', fontsize='16')
    plt.title(f'{title}', fontsize='16')
    # custom tick labels
    ax.xaxis.set_ticks(np.arange(
        start_km, (start_km*1000 + 0.05*(len(error_list)-0.9))/1000, 0.1))
    # Create a formatter function for tick labels in [m]
    def multiply_by_1000(x, pos):
        return f'{x * 1000:.0f}'
    # Set custom formatter for x-axis ticks
    formatter = FuncFormatter(multiply_by_1000)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xticks(fontsize='16')
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

    cbar = fig.colorbar(c, ax=ax) # , spacing='uniform'
    
    tick_locs = np.linspace(int(df['Label'].min()),
                            int(df['Label'].max()+1),
                            int(2 * n_clusters + 1))[1::2]
    
    tick_label = np.arange(int(df['Label'].min()),
                                int(df['Label'].max() + 1))
    
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(tick_label)
    
    cbar.ax.set_ylim(df['Label'].min(), int(df['Label'].max()+1))
    
    ax.legend(['Reconstruction Error', 'Threshold',
               'Threshold adjusted'],
              fontsize='16',
              loc='best')

    plt.tight_layout()
    plt.savefig(fr'02_Results\{tunnel}\02_Plots\02_errors_vs_CLASSes\{title}_{file_name}.png', dpi=300)

    return df

df_Error_test_1 = plot_reconstruction_error(error_test1,
                                            test_dataloader1,
                                            sequence_length,
                                            start_test_1,
                                            'Test Dataset 1',
                                            file_name,
                                            threshold_all,
                                            adjusted_threshold_all
                                            )

df_Error_test_2 = plot_reconstruction_error(error_test2,
                                            test_dataloader2,
                                            sequence_length,
                                            start_test_2,
                                            'Test Dataset 2',
                                            file_name,
                                            threshold_all,
                                            adjusted_threshold_all
                                            )

df_Error_test_3 = plot_reconstruction_error(error_test3,
                                            test_dataloader3,
                                            sequence_length,
                                            start_test_3,
                                            'Test Dataset 3',
                                            file_name,
                                            threshold_all,
                                            adjusted_threshold_all
                                            )

