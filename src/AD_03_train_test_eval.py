# -*- coding: utf-8 -*-
"""
TBM Operational Data-Driven Anomaly Detection in Hard Rock Excavations

Script for training, testing, and evaluating the VAE-based anomaly detection model.

@author: Paul Unterlaß / Mario Wölflingseder
"""

# =============================================================================
# Imports
# =============================================================================
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter, ticker
from statsmodels.stats import stattools

from VAE_04_model import VAE
from VAE_02_preprocessor_for_VAE import TBMDataset

# =============================================================================
# Hyperparameters
# =============================================================================
tunnel = 'TBM_A'  # Options: 'TBM_B', 'Synth_TBM_A_B', 'Synth_TBM_B_A'

if tunnel in ['Synth_TBM_A', 'Synth_TBM_A_B', 'Synth_TBM_B_A']:
    input_size = 5  # Number of input features
else:
    input_size = 8

hidden_size_encoder = 30  # Hidden state size for LSTM encoder
hidden_size_decoder = 30  # Hidden state size for LSTM decoder
latent_size = 3  # Latent space dimension
batch_size = 64
num_epochs = 50
learning_rate = 0.0002
sequence_length = 100
beta = 0.001  # Weight for KL-divergence loss

# =============================================================================
# Load Datasets and Create DataLoaders
# =============================================================================
if tunnel == 'TBM_B':
    CLASS = 5
    start_val = 3.5
    end_val = 4.5
    start_test_1 = 2.5
    end_test_1 = 2.8
    start_test_2 = 3.0
    end_test_2 = 3.3
    start_test_3 = 1.5
    end_test_3 = 1.8
    interval = 0.03

elif tunnel == 'Synth_TBM_A_B':
    CLASS = 4
    start_val = 0
    end_val = 0.25
    start_test_1 = 2.4
    end_test_1 = 2.8
    start_test_2 = 4.9
    end_test_2 = 5.25
    start_test_3 = 5.9
    end_test_3 = 6.2
    interval = 0.03

elif tunnel == 'Synth_TBM_A':
    CLASS = 3
    start_val = 0
    end_val = 0.25
    start_test_1 = 3.5
    end_test_1 = 4.5
    start_test_2 = 5.0
    end_test_2 = 6.0
    start_test_3 = 9.5
    end_test_3 = 10.5
    interval = 0.05

elif tunnel == 'TBM_A':
    CLASS = 3
    start_val = 10.5
    end_val = 11.5
    start_test_1 = 3.5
    end_test_1 = 4.5
    start_test_2 = 5.0
    end_test_2 = 6.0
    start_test_3 = 9.5
    end_test_3 = 10.5
    interval = 0.05

else:
    raise ValueError("Tunnel type not defined.")

# Define file naming convention
file_name = (
    f'{tunnel}_CLASS{CLASS}_seq{sequence_length}_beta{beta}_ep{num_epochs}_'
    f'hse{hidden_size_encoder}_hsd{hidden_size_decoder}_ls{latent_size}'
)

# Load training dataset
train_dataset = torch.load(
    f'01_data/{tunnel}/datasets/{tunnel}_{CLASS}_train_dataset_seq{sequence_length}.pt'
)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# Load test datasets

test_dataset1 = torch.load(
    f'01_data/{tunnel}/datasets/{tunnel}_{CLASS}_test_dataset1_seq{sequence_length}_{start_test_1}-{end_test_1}.pt'
)
test_dataloader1 = DataLoader(test_dataset1, batch_size=batch_size, shuffle=False)


test_dataset2 = torch.load(
    f'01_data/{tunnel}/datasets/{tunnel}_{CLASS}_test_dataset2_seq{sequence_length}_{start_test_2}-{end_test_2}.pt'
)
test_dataloader2 = DataLoader(test_dataset2, batch_size=batch_size, shuffle=False)


test_dataset3 = torch.load(
    f'01_data/{tunnel}/datasets/{tunnel}_{CLASS}_test_dataset3_seq{sequence_length}_{start_test_3}-{end_test_3}.pt'
)
test_dataloader3 = DataLoader(test_dataset3, batch_size=batch_size, shuffle=False)

# Load validation dataset
val_dataset = torch.load(
    f'01_data/{tunnel}/datasets/{tunnel}_{CLASS}_val_dataset_seq{sequence_length}_{start_val}-{end_val}.pt'
)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# =============================================================================
# Model Initialization
# =============================================================================

# Create model instance
vae = VAE(
    input_size,
    hidden_size_encoder,
    hidden_size_decoder,
    latent_size,
    sequence_length
)

# Uncomment to load a pre-trained model
# model = torch.load(f'02_Results/{tunnel}/01_Models/{file_name}.pth')
# history = torch.load(f'02_Results/{tunnel}/01_Models/history of {file_name}.pt')

# =============================================================================
# Training Function
# =============================================================================

def train_vae(
    vae,
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
    """
    Trains the VAE model, evaluates it on test and validation sets, and saves the best model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device)

    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction='sum').to(device)
    
    best_eval_loss = float("inf")
    
    history = {
        "train_rec": [], "train_kl": [],
        "test_rec": [], "test_kl": [],
        "val_rec": [], "val_kl": []
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        
        losses = {
            "train_rec": [], "train_kl": [],
            "test_rec": [], "test_kl": [],
            "val_rec": [], "val_kl": []
        }
        
        vae.train()
        
        for input_data, _ in tqdm(train_dataloader):
            input_data = input_data.to(device).float()
            reconstructed_data, mu, logvar = vae(input_data)

            reconstruction_loss = criterion(reconstructed_data, input_data)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = reconstruction_loss + beta * kl_div
            
            losses["train_rec"].append(reconstruction_loss.item())
            losses["train_kl"].append(beta * kl_div.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        vae.eval()
        
        with torch.no_grad():
            for dataloader, prefix in zip(
                [test_dataloader1, test_dataloader2, test_dataloader3, val_dataloader],
                ["test", "test", "test", "val"]
            ):
                for input_data, _ in tqdm(dataloader):
                    input_data = input_data.to(device).float()
                    reconstructed_data, mu, logvar = vae(input_data)
                    
                    reconstruction_loss = criterion(reconstructed_data, input_data)
                    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    
                    losses[f"{prefix}_rec"].append(reconstruction_loss.item())
                    losses[f"{prefix}_kl"].append(beta * kl_div.item())
        
        for key in losses:
            epoch_loss = np.mean(losses[key])
            history[key].append(epoch_loss)
            print(f'{key}: {epoch_loss:.8f}')
        
        eval_loss = np.mean(losses['val_rec']) + np.mean(losses['val_kl'])
        
        if best_eval_loss > eval_loss:
            best_eval_loss = eval_loss
            torch.save(vae, f'02_Results/{tunnel}/01_Models/{file_name}.pth')
            torch.save(history, f'02_Results/{tunnel}/01_Models/history of {file_name}.pt')
    
    print(f'TRAINING COMPLETE: {file_name}.pth')
    
    plt.figure()
    plt.plot(history['train_rec'], label='train')
    plt.plot(history['test_rec'], label='test')
    plt.plot(history['val_rec'], label='validation')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xlim(0, num_epochs)
    plt.legend()
    plt.title('Loss over training epochs')
    plt.savefig(f'02_Results/{tunnel}/02_Plots/01_learning_curve/Learning Curve_{file_name}.png')
    plt.show()
    
    return vae.eval(), history

# =============================================================================
# Train the VAE and Plot Learning Curve
# =============================================================================
model, history = train_vae(
    vae,
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

# =============================================================================
# Test the Model
# =============================================================================

def test_vae(vae, dataloader):
    """
    Evaluates the trained VAE model and computes reconstruction errors.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device)

    error_list = []
    vae.eval()
    
    with torch.no_grad():
        for input_data, _ in dataloader:
            input_data = input_data.to(device).float()
            
            # Forward pass
            reconstructed_data, mu, logvar = vae(input_data)
            
            # Compute reconstruction error
            error = torch.mean((reconstructed_data - input_data) ** 2, dim=(1, 2)).cpu()
            error_list.extend(error.numpy().flatten())
    
    return error_list, mu, logvar

# Run model testing on different test sets
error_test1, lat_mu_test1, lat_logvar_test1 = test_vae(model, test_dataloader1)
error_test2, lat_mu_test2, lat_logvar_test2 = test_vae(model, test_dataloader2)
error_test3, lat_mu_test3, lat_logvar_test3 = test_vae(model, test_dataloader3)
error_test_sum = error_test1 + error_test2 + error_test3

# =============================================================================
# Threshold Calculation
# =============================================================================

def threshold(error_test, title):
    """
    Computes threshold values for anomaly detection using IQR and skewness adjustment.
    """
    error_test = np.array(error_test)
    print(f'Highest error value: {error_test.max()}')
    
    quantile = np.quantile(error_test, 0.99)
    print(f'99% quantile: {quantile}')
    
    q1 = np.percentile(error_test, 25)
    q3 = np.percentile(error_test, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    print(f'Upper boundary: {upper_bound}')
    
    # Compute medcouple (skewness measure)
    mc = statsmodels.stats.stattools.medcouple(error_test, axis=0)
    print(f'Medcouple: {mc}')
    
    if mc < 0:
        adjusted_lower_bound = q1 - 1.5 * np.exp(-3 * mc) * iqr
        adjusted_upper_bound = q3 + 1.5 * np.exp(4 * mc) * iqr
    else:
        adjusted_lower_bound = q1 - 1.5 * np.exp(-4 * mc) * iqr
        adjusted_upper_bound = q3 + 1.5 * np.exp(3 * mc) * iqr
    
    print(f'Skewness adjusted lower boundary: {adjusted_lower_bound}')
    print(f'Skewness adjusted upper boundary: {adjusted_upper_bound}')
    
    # Plot error distribution
    sns.set(style="ticks")
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, figsize=(15, 10), gridspec_kw={"height_ratios": (.15, .85)})
    
    if mc < 0:
        whis = q3 + 1.5 * np.exp(4 * mc)
    else:
        whis = q3 + 1.5 * np.exp(3 * mc)
    
    sns.boxplot(x=error_test, ax=ax_box, orient='h', whis=whis, color='lightgrey', linewidth=3)
    sns.histplot(error_test, ax=ax_hist, kde=True)
    
    if error_test.max() > 0.1:
        x_max = adjusted_upper_bound + 0.1
    elif adjusted_upper_bound < 0.5:
        x_max = adjusted_upper_bound + 0.005
    else:
        x_max = error_test.max()
    
    plt.xlim(0, 0.06)
    plt.ylabel('Instances', fontsize=16)
    plt.xlabel('Reconstruction Error [-]', fontsize=16)
    plt.xticks(rotation=90, fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    
    plt.subplots_adjust(top=0.95)
    plt.suptitle(title, fontsize=16)
    plt.axvline(upper_bound, c='black', ls='--', label='Upper boxplot boundary')
    plt.axvline(adjusted_upper_bound, c='red', ls='--', label='Skewness adjusted upper boundary')
    ax_box.set(yticks=[])
    sns.despine(ax=ax_hist)
    sns.despine(ax=ax_box, left=True)
    ax_hist.legend(loc='upper right', facecolor='white', edgecolor='black', framealpha=1, prop={'size': 16})
    
    return adjusted_upper_bound, upper_bound

# Compute thresholds for different test sets
adjusted_threshold_test1, threshold_test1 = threshold(error_test1, 'Test Set 1')
adjusted_threshold_test2, threshold_test2 = threshold(error_test2, 'Test Set 2')
adjusted_threshold_test3, threshold_test3 = threshold(error_test3, 'Test Set 3')
adjusted_threshold_all, threshold_all = threshold(error_test_sum, 'All Test Sets')

# =============================================================================
# Function to Plot Reconstruction Errors
# =============================================================================
def plot_reconstruction_error(error_list,
                              dataloader,
                              sequence_length,
                              start_km,
                              title,
                              file_name,
                              threshold,
                              adjusted_threshold,
                              interval):
    """
    This function plots the reconstruction errors and thresholds for a given dataset.
    
    Parameters:
    - error_list: List of reconstruction errors
    - dataloader: Dataloader object that provides the dataset
    - sequence_length: Length of the sequence used for analysis
    - start_km: Starting kilometer of the test section
    - title: Title of the plot
    - file_name: The name of the output file for saving the plot
    - threshold: Threshold value for comparison
    - adjusted_threshold: Threshold adjusted for skewness
    - interval: Distance interval between points in kilometers
    
    Returns:
    - df: DataFrame containing the processed data for plotting
    """
    
    # Create a list of labels from the input data
    label_list = []
    for _, label in dataloader:
        for label in label:
            label = label.numpy()  # Convert label to a numpy array
            label = float(label)  # Convert label to a float
            label_list.append(label)

    # Create a list of threshold values matching the length of error_list
    threshold_for_plot = [threshold] * len(error_list)
    adjusted_threshold_for_plot = [adjusted_threshold] * len(error_list)

    # Create a DataFrame with the error, threshold, adjusted threshold, and labels
    df = pd.DataFrame(
        list(zip(error_list, threshold_for_plot, adjusted_threshold_for_plot, label_list)),
        columns=['Error', 'Threshold', 'Adjusted Threshold', 'Label']
    )

    # Add tunnel distance to the DataFrame
    df['Tunnel Distance [km]'] = np.arange(start_km, 
                                            (start_km * 1000 + interval * (len(error_list) - 0.9)) / 1000, 
                                            interval / 1000)

    # Shift error values to center them around half the sequence length
    df['Error'] = df['Error'].shift(int(sequence_length / 2))

    # Cut the DataFrame to remove the first part (half of sequence length)
    df = df[int(sequence_length / 2):]
    df = df.reset_index(drop=True)

    # Plotting the reconstruction errors
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.scatter(df['Tunnel Distance [km]'], df['Error'], s=2, c='black', label='Reconstruction Error')
    ax.set_xlim(start_km + interval * int(sequence_length / 2) / 1000,
                start_km + interval * len(error_list) / 1000)
    ax.set_ylim(0, 0.08)

    # Plot the threshold and adjusted threshold
    ax.plot(df['Threshold'], color="black", linewidth=2, linestyle='--', label='Threshold')
    ax.plot(df['Adjusted Threshold'], color='red', linewidth=2, linestyle='--', label='Threshold Adjusted')

    # Labeling axes and adding title
    plt.ylabel('Reconstruction Error', fontsize='16')
    plt.xlabel('Tunnel Distance [m]', fontsize='16')
    plt.title(f'{title}', fontsize='16')

    # Custom tick labels for the x-axis (in meters)
    def multiply_by_1000(x, pos):
        return f'{x * 1000:.0f}'

    formatter = FuncFormatter(multiply_by_1000)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_ticks(np.arange(start_km, 
                                  (start_km * 1000 + interval * (len(error_list) - 0.9)) / 1000, 0.1))
    plt.xticks(fontsize='16')
    plt.yticks(fontsize='16')

    # Format the y-axis ticks to display two decimal places
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    # Determine the number of clusters
    n_clusters = df['Label'].max() - df['Label'].min() + 1

    # Set the colormap based on the number of clusters
    if n_clusters == 6:
        cmap = mpl.colors.ListedColormap(['green', 'greenyellow', 'gold', 'orange', 'darkorange', 'red'])
        cmap = cmap.with_extremes(over='red', under='green')
    elif n_clusters == 5:
        cmap = mpl.colors.ListedColormap(['green', 'greenyellow', 'gold', 'orange', 'darkorange'])
        cmap = cmap.with_extremes(over='red', under='green')
    elif n_clusters == 4:
        if tunnel == 'Synth_TBM_B_A' or tunnel == 'Synth_TBM_A' or tunnel == 'TBM_A':
            cmap = mpl.colors.ListedColormap(['green', 'gold', 'orange', 'red'])
        else:
            cmap = mpl.colors.ListedColormap(['green', 'greenyellow', 'gold', 'orange'])
        cmap = cmap.with_extremes(over='red', under='green')
    elif n_clusters == 3:
        cmap = mpl.colors.ListedColormap(['green', 'gold', 'red'])
        cmap = cmap.with_extremes(over='red', under='green')
    elif n_clusters == 2:
        if tunnel == 'TBM_B':
            cmap = mpl.colors.ListedColormap(['green', 'red'])
        else:
            cmap = mpl.colors.ListedColormap(['green', 'gold'])
    else:
        print('Color map not suited for the data to be plotted')

    # Add color mapping to the plot
    c = ax.pcolorfast(ax.get_xlim(), ax.get_ylim(), 
                      df['Label'].values[np.newaxis], cmap=cmap,
                      vmin=df['Label'].min(), vmax=df['Label'].max() + 1, alpha=0.5)

    # Create and format the colorbar
    cbar = fig.colorbar(c, ax=ax)
    tick_locs = np.linspace(int(df['Label'].min()), int(df['Label'].max() + 1), int(2 * n_clusters + 1))[1::2]
    tick_label = np.arange(int(df['Label'].min()), int(df['Label'].max() + 1))
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(tick_label)
    cbar.ax.set_ylim(df['Label'].min(), int(df['Label'].max() + 1))

    # Add legend
    ax.legend(fontsize='16', loc='best')

    # Ensure the layout is tight
    plt.tight_layout()

    # Uncomment below line to save the plot
    # plt.savefig(fr'02_Results\{tunnel}\02_Plots\02_errors_vs_CLASSes\{title}_{file_name}.png', dpi=300)

    return df

# Example function calls for three test datasets
df_Error_test_1 = plot_reconstruction_error(error_test1,
                                            test_dataloader1,
                                            sequence_length,
                                            start_test_1,
                                            'Test Dataset 1',
                                            file_name,
                                            threshold_test1,
                                            adjusted_threshold_test1,
                                            interval)

df_Error_test_2 = plot_reconstruction_error(error_test2,
                                            test_dataloader2,
                                            sequence_length,
                                            start_test_2,
                                            'Test Dataset 2',
                                            file_name,
                                            threshold_test2,
                                            adjusted_threshold_test2,
                                            interval)

df_Error_test_3 = plot_reconstruction_error(error_test3,
                                            test_dataloader3,
                                            sequence_length,
                                            start_test_3,
                                            'Test Dataset 3',
                                            file_name,
                                            threshold_test3,
                                            adjusted_threshold_test3,
                                            interval)

# =============================================================================
# save results
# =============================================================================

def save_df(error_list,
            dataloader,
            sequence_length,
            start_km,
            title,
            file_name,
            threshold,
            adjusted_threshold,
            tunnel_name):
    """
    Processes reconstruction errors and saves the resulting DataFrame to a
    CSV file.
    
    Parameters:
        error_list: List of reconstruction errors.
        dataloader: Dataloader containing the labels for each data point.
        sequence_length: Length of the sequences used in reconstruction.
        start_km: Starting kilometer for x-axis.
        title: Plot title.
        file_name: Name of the output file.
        threshold: Original threshold.
        adjusted_threshold: Adjusted threshold for anomaly detection.
        tunnel_name: Name of the tunnel for labeling.
    """

    # Create a list of labels from the dataloader
    label_list = []
    for _, label in dataloader:
        for lbl in label:
            label_list.append(float(lbl.numpy()))

    # Prepare threshold lists for the plot
    threshold_for_plot = [threshold] * len(error_list)
    adjusted_threshold_for_plot = [adjusted_threshold] * len(error_list)

    # Create a DataFrame
    df = pd.DataFrame({
        'Error': error_list,
        'Threshold': threshold_for_plot,
        'Adjusted Threshold': adjusted_threshold_for_plot,
        'Label': label_list
    })

    # Generate Tunnel Distance for x-axis
    df['Tunnel Distance [km]'] = np.linspace(
        start_km, 
        start_km + interval * len(error_list) / 1000, 
        len(error_list)
    )

    # Shift errors to align with sequence midpoint
    df['Error'] = df['Error'].shift(int(sequence_length / 2))

    # Drop the first rows corresponding to the shift
    df = df[int(sequence_length / 2):].reset_index(drop=True)

    # Identify anomalies based on the adjusted threshold
    df['Anomaly'] = (df['Error'] >= adjusted_threshold).astype(int)

    # Save the DataFrame to a CSV file
    df.to_csv(fr'02_Results\{tunnel}\02_Plots\07_VAE\{title}_{file_name}.csv', index=False)
    
    return df

df_Error_test_1_anomalies = save_df(error_test1,
                                    test_dataloader1,
                                    sequence_length,
                                    start_test_1,
                                    'Test Dataset 1',
                                    file_name,
                                    threshold_test1,
                                    adjusted_threshold_test1,
                                    tunnel)

df_Error_test_2_anomalies = save_df(error_test2,
                                    test_dataloader2,
                                    sequence_length,
                                    start_test_2,
                                    'Test Dataset 2',
                                    file_name,
                                    threshold_test2,
                                    adjusted_threshold_all,
                                    tunnel)

df_Error_test_3_anomalies = save_df(error_test3,
                                    test_dataloader3,
                                    sequence_length,
                                    start_test_3,
                                    'Test Dataset 3',
                                    file_name,
                                    threshold_test3,
                                    adjusted_threshold_test3,
                                    tunnel)

