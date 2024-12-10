 # -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 2024

@author: unterlass/wölflingseder
"""
'''
pre-processing for VAE based anomaly detection
'''

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.stats.stattools
import seaborn as sns

from VAE_05_model import VAE
from VAE_05_model import train_vae
from VAE_05_model import test_vae
from VAE_05_model import detect_anomalies
from VAE_05_model import plot_reconstructed_seq
from VAE_05_model import plot_reconstruction_error
from VAE_05_model import plot_original_data
from VAE_03_preprocessor_3 import TBMDataset

# =============================================================================
# Hyperparameters
# =============================================================================
TUNNEL = 'Synth_BBT_UT' #'Synth_BBT' #'UT'

if TUNNEL == 'Synth_BBT' or 'Synth_BBT_UT':    
    INPUT_SIZE = 5  # Size of input features
else:
    INPUT_SIZE = 8

HIDDEN_SIZE_ENCODER = 30  # Size of hidden state in LSTM layers of encoder
HIDDEN_SIZE_DECODER = 30 # Size of hidden state in LSTM layers of decoder
LATENT_SIZE = 3  # Size of the latent representation
BATCH_SIZE = 64 # 32
NUM_EPOCHS =  50
LEARNING_RATE = 0.0002
SEQUENCE_LENGTH = 100
BETA = 0.001

# =============================================================================
# creating dataloader
# =============================================================================
if TUNNEL == 'UT':
    CLASS = 4
    START_VAL = 3.5
    END_VAL = 4.5
    START_TEST1 = 2.4
    END_TEST1 = 2.8
    START_TEST2 = 4.9
    END_TEST2 = 5.25
    START_TEST3 = 5.9
    END_TEST3 = 6.2
    
elif TUNNEL == 'Synth_BBT_UT':
    CLASS = 4
    START_VAL = 0
    END_VAL = 0.25
    START_TEST1 = 2.4
    END_TEST1 = 2.8
    START_TEST2 = 4.9
    END_TEST2 = 5.25
    START_TEST3 = 5.9
    END_TEST3 = 6.2
    
elif TUNNEL == 'Synth_BBT':
    CLASS = 3
    START_VAL = 0
    END_VAL = 0.25
    START_TEST1 = 3.5
    END_TEST1 = 4.5
    START_TEST2 = 5.0
    END_TEST2 = 6.0
    START_TEST3 = 9.5
    END_TEST3 = 10.5
    
elif TUNNEL == 'BBT':
    CLASS = 3
    START_VAL = 10.5
    END_VAL = 11.5
    START_TEST1 = 3.5
    END_TEST1 = 4.5
    START_TEST2 = 5.0
    END_TEST2 = 6.0
    START_TEST3 = 9.5
    END_TEST3 = 10.5

else:
    print('tunnel not defined')
    
FILE_NAME = f'{TUNNEL}_CLASS{CLASS}_seq{SEQUENCE_LENGTH}_beta{BETA}_ep{NUM_EPOCHS}_hse{HIDDEN_SIZE_ENCODER}_hsd{HIDDEN_SIZE_DECODER}_ls{LATENT_SIZE}'

# train
train_dataset = torch.load(
    f'01_data/{TUNNEL}/datasets/{TUNNEL}_{CLASS}_train_dataset_seq{SEQUENCE_LENGTH}.pt'
    )

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
    )

# test 1-3
test_dataset1 = torch.load(
    f'01_data/{TUNNEL}/datasets/{TUNNEL}_{CLASS}_test_dataset1_seq{SEQUENCE_LENGTH}_{START_TEST1}-{END_TEST1}.pt')

test_dataloader1 = DataLoader(
    test_dataset1,
    batch_size=BATCH_SIZE,
    shuffle=False
    )

test_dataset2 = torch.load(
    f'01_data/{TUNNEL}/datasets/{TUNNEL}_{CLASS}_test_dataset2_seq{SEQUENCE_LENGTH}_{START_TEST2}-{END_TEST2}.pt')

test_dataloader2 = DataLoader(
    test_dataset2,
    batch_size=BATCH_SIZE,
    shuffle=False
    )

test_dataset3 = torch.load(
    f'01_data/{TUNNEL}/datasets/{TUNNEL}_{CLASS}_test_dataset3_seq{SEQUENCE_LENGTH}_{START_TEST3}-{END_TEST3}.pt')

test_dataloader3 = DataLoader(
    test_dataset3,
    batch_size=BATCH_SIZE,
    shuffle=False)

# validation
val_dataset = torch.load(
    f'01_data/{TUNNEL}/datasets/{TUNNEL}_{CLASS}_val_dataset_seq{SEQUENCE_LENGTH}_{START_VAL}-{END_VAL}.pt'
    )

val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
    )

# dataframe only for plotting original data
df_pen_tor = pd.read_csv('spec. penetration and torque ratio.csv')

# =============================================================================
# training
# =============================================================================

# Create an instance of the VAE
vae = VAE(INPUT_SIZE,
          HIDDEN_SIZE_ENCODER,
          HIDDEN_SIZE_DECODER,
          LATENT_SIZE,
          SEQUENCE_LENGTH
          )

# # load model
# model = torch.load('withoutGI4_seq100_beta0.001_ep50.pth')
# history = torch.load('history of withoutGI4_seq100_beta0.001_ep50.pth.pt')

# Train the VAE / plot learning curve 
model, history = train_vae(vae,
                           train_dataloader,
                           test_dataloader1,
                           test_dataloader2,
                           test_dataloader3,
                           val_dataloader,
                           NUM_EPOCHS,
                           LEARNING_RATE,
                           BETA,
                           FILE_NAME
                           )

# # load best model after training model
# model = torch.load(f'02_Results/Synth_BBT/01_Models/{FILE_NAME}' + '.pth')
# history = torch.load(f'02_Results/Synth_BBT/01_Models/history of {FILE_NAME}.pt')

# create reconstruction errors
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
    print(error_test.max())
    
    quantile = np.quantile(error_test, 0.99)
    print('99% quantil:', quantile)

    q1 = np.percentile(error_test, 25)
    q3 = np.percentile(error_test, 75)
    iqr = q3 - q1

    boundary_low = q1 - 1.5 * iqr
    boundary_up = q3 + 1.5 * iqr

    print('boundary up:', boundary_up)

    mc = statsmodels.stats.stattools.medcouple(error_test, axis=0)

    skewed_boundary_low = q1 - 1.5*np.exp(-4*mc)*iqr
    skewed_boundary_up = q3 + 1.5*np.exp(3*mc)*iqr

    print('low:', 1.5*np.exp(3*mc))

    print('skewed boundary up:', skewed_boundary_up)
    print('skewed boundary low:', skewed_boundary_low)
    
    # plot error
    sns.set(style="ticks")

    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, figsize=(15, 10),
                                        gridspec_kw={"height_ratios": (.15, .85)})

    sns.boxplot(error_test,
                ax=ax_box,
                orient='h',
                whis=(q3 + 1.5*np.exp(3*mc)),
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
        x_max = skewed_boundary_up + 0.005
        
    elif skewed_boundary_up < 0.5:
        x_max = skewed_boundary_up + 0.5
        
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
    
    ax_box.set(yticks=[])
    sns.despine(ax=ax_hist)
    sns.despine(ax=ax_box, left=True)
    plt.savefig(fr'02_Results\{TUNNEL}\02_Plots\03_threshold\{title}_{FILE_NAME}.png')

    
    return skewed_boundary_up

threshold_test1 = threshold(error_test1, 'test_set_1')
threshold_test2 = threshold(error_test2, 'test_set_2')
threshold_test3 = threshold(error_test3, 'test_set_3')
threshold_all = threshold(error_test_sum, 'all_test_sets')

# plots reconstruction error over sequence index

df_Error_test_1 = plot_reconstruction_error(error_test1,
                                            test_dataloader1,
                                            SEQUENCE_LENGTH,
                                            START_TEST1,
                                            'Test Dataset 1',
                                            FILE_NAME,
                                            threshold=threshold_all
                                            )

df_Error_test_2 = plot_reconstruction_error(error_test2,
                                            test_dataloader2,
                                            SEQUENCE_LENGTH,
                                            START_TEST2,
                                            'Test Dataset 2',
                                            FILE_NAME,
                                            threshold=threshold_all
                                            )

df_Error_test_3 = plot_reconstruction_error(error_test3,
                                            test_dataloader3,
                                            SEQUENCE_LENGTH,
                                            START_TEST3,
                                            'Test Dataset 3',
                                            FILE_NAME,
                                            threshold=threshold_all
                                            )

# for checking if plot is working correctly
df_Error_test_1['Label'] = df_Error_test_1['Label']/max(df_Error_test_1['Label'])
df_Error_test_1['Error'] = df_Error_test_1['Error']/max(df_Error_test_1['Error'])
df_Error_test_1.plot(y=['Label', 'Error'])

df_section = plot_original_data(df_pen_tor,
                                START_TEST1,
                                END_TEST1,
                                'spec. penetration [mm/rot/MN]',
                                'Test Dataset 1'
                                )

df_section = plot_original_data(df_pen_tor,
                                START_TEST1,
                                END_TEST1,
                                'torque ratio', 
                                'Test Dataset 2'
                                )


# # plots examples of reconstructed sequences for train and test dataset
# reconstructed_data_train, input_data_train = plot_reconstructed_seq(model, train_dataloader, 0)
# reconstructed_data_test, input_data_test = plot_reconstructed_seq(model, test_dataloader, 0)

# # anomaly detection on train and test dataset
# anomaly_count_train, no_anomaly_count_test = detect_anomalies(model, train_dataloader, THRESHOLD)  
# anomaly_count_train, no_anomaly_count_test = detect_anomalies(model, test_dataloader, THRESHOLD)

# # plots multiple examples of reconstructed sequences
# spacing_m = 100 # in meter
# start_km = 0.1 # in km
# num_examples = 5
# examples = []
# for i in range(num_examples):
#     examples.append(round(start_km+i*spacing_m/1000, 10))

# # finds sequences with lowest and highest reconstruction loss in the training dataset
# examples = [error_train.index(min(error_train))/20000, error_train.index(max(error_train))/20000]
# print(f'train min = {min(error_train)}; train max = {max(error_train)}')

# # plots sequences with lowest and highest reconstruction loss in the training dataset
# for example in examples:
#     plot_reconstructed_seq(model, train_dataloader, example)

# # finds sequences with lowest and highest reconstruction loss in the test dataset
# examples = [error_test.index(min(error_test))/20000, error_test.index(max(error_test))/20000]
# print(f'test min = {min(error_test)}; test max = {max(error_test)}')

# # plots sequences with lowest and highest reconstruction loss in the test dataset
# for example in examples:
#     plot_reconstructed_seq(model, test_dataloader, example)

# plots histogram of reconstruction 

# plt.hist(error_train, bins=1000)
# plt.hist(error_test, bins=3000)
# plt.xlim(xmin=0, xmax = 0.01)
# plt.title('Error Histogram of Test Dataset')
# plt.ylabel('INSTANCES')
# plt.xlabel('RECONSTRUCTION ERROR')
