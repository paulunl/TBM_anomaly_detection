# -*- coding: utf-8 -*-
"""
TBM Operational Data-Driven Anomaly Detection in Hard Rock Excavations

Pre-processing routine for the (V)AE-based anomaly detection model.

@author: Paul Unterlaß / Mario Wölflingseder
"""

# =============================================================================
# Imports
# =============================================================================

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
# Load dataset
# =============================================================================

tunnel = 'TBM_B'  # Options: 'TBM_A', 'Synth_TBM_A_B', 'Synth_TBM_B_A'
Class = 4  # Threshold for classification filtering

def load_and_preprocess_data(tunnel):
    """
    Loads TBM dataset, applies filtering, and standardizes column names.
    """
    df = pd.read_parquet('path_to_data')  # Replace with actual path
    
    if tunnel in ['TBM_B', 'Synth_TBM_A_B']:
        df['Class'] -= 1
        df = df[(df['Torque cutterhead [MNm]'] <= 10.2) &
                (df['Total advance force [kN]'].between(4000, 27000)) &
                (df['Penetration [mm/rot]'] >= 0.1)]
    
    elif tunnel in ['TBM_A', 'Synth_TBM_A', 'Synth_TBM_B_A']:
        df = df[(df['Torque cutterhead [MNm]'] <= 4.5) &
                (df['Total advance force [kN]'].between(2000, 17500)) &
                (df['Penetration [mm/rot]'] >= 0.1)]
        df = df[df['Tunnel Distance [m]'] > 1000]  # Remove first km
        df.rename(columns={'GI': 'Class',
                           'Speed cutterhead [rpm]': 'Speed cutterhead for display [rpm]'}, inplace=True)
    
    df.reset_index(drop=True, inplace=True)
    return df

df = load_and_preprocess_data(tunnel)

# =============================================================================
# Define dataset sections and sequence parameters
# =============================================================================

def define_dataset_splits(tunnel):
    """
    Defines dataset columns, validation, and test splits based on tunnel type.
    """
    base_columns = ['Penetration [mm/rot]', 'Speed cutterhead for display [rpm]',
                    'Torque cutterhead [MNm]', 'Total advance force [kN]', 'torque ratio']
    
    additional_columns = ['Advance speed [mm/min]', 'Pressure advance cylinder bottom side [bar]', 'spec. penetration [mm/rot/MN]']
    
    if tunnel in ['TBM_A', 'TBM_B']:
        columns = base_columns + additional_columns
    else:
        columns = base_columns
    
    target_column = ['Class']
    seq_size = 100
    
    # Define dataset splits (values in km)
    splits = {
        'TBM_A': (10.5, 11.5, 3.5, 4.5, 5.0, 6.0, 9.5, 10.5),
        'TBM_B': (3.5, 4.5, 2.5, 2.8, 3.0, 3.3, 1.5, 1.8),
        'Synth_TBM_A': (0, 0.25, 3.5, 4.5, 5.0, 6.0, 9.5, 10.5),
        'Synth_TBM_A_B': (0, 0.25, 2.4, 2.8, 4.9, 5.25, 5.9, 6.2),
        'Synth_TBM_B_A': (0, 0.25, 3.5, 4.5, 5.0, 6.0, 9.5, 10.5)
    }
    
    return columns, target_column, seq_size, splits[tunnel]

columns, target_column, seq_size, (val_start, val_end, test_start1, test_end1, test_start2, test_end2, test_start3, test_end3) = define_dataset_splits(tunnel)

# =============================================================================
# Normalize Data
# =============================================================================

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(df[columns])

def normalize(df):
    return pd.DataFrame(scaler.transform(df[columns]), index=df.index, columns=columns)

train_df = normalize(df)

# =============================================================================
# Create Sequences for LSTM
# =============================================================================

def create_sequences(dataset, target_df, seq_size):
    """Generates sequences of specified length from the dataset."""
    sequences = []
    
    for i in range(len(dataset) - seq_size):
        sequence = dataset[i:i+seq_size]
        label_pos = i + seq_size - 1
        label = target_df.iloc[label_pos]
        sequences.append((sequence, label))
    
    return sequences

train_seq = create_sequences(train_df, train_df[target_column], seq_size)

# =============================================================================
# Torch Dataset Class
# =============================================================================

class TBMDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        return torch.tensor(sequence.to_numpy()), torch.tensor(label)

# Convert to PyTorch dataset
train_data = TBMDataset(train_seq)

# =============================================================================
# Save Processed Dataset
# =============================================================================

def save_dataset(dataset, filename):
    torch.save(dataset, f'01_data/{tunnel}/datasets/' + filename)

save_dataset(train_data, f'{tunnel}_train.pt')

print("Preprocessing completed and dataset saved.")
