 # -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 2024

@author: unterlass/wölflingseder
"""
'''
pre-processing for (V)AE based anomaly detection
'''

# =============================================================================
# imports
# =============================================================================

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
# load the dataset
# =============================================================================

tunnel = 'UT' # 'BBT' # 'Synth_BBT_UT'
Class = 4

if tunnel == 'UT':
    df = pd.read_parquet(fr'D:\02_Research\01_Unterlass\05_Anomaly_detection\01_data\{tunnel}\01_TBM_data_preprocessed_Qclass.gzip')
    # df = pd.read_parquet(fr'E:\Paul Unterlass\Anomaly_detection\01_data\{tunnel}\01_TBM_data_preprocessed_Qclass.gzip')
    df['Class'] = df['Class'] - 1
    
    # hard drop of outliers which lie beyond the machine limits
    df.drop(df[df['Torque cutterhead [MNm]'] > 10.2].index, inplace=True)
    df.drop(df[df['Total advance force [kN]'] > 27000].index, inplace=True)
    df.drop(df[df['Total advance force [kN]'] < 4000].index, inplace=True)
    df.drop(df[df['Penetration [mm/rot]'] < 0.1].index, inplace=True)
    df.reset_index(inplace=True, drop=True)

elif tunnel == 'BBT':
    # df = pd.read_parquet(fr'E:\Paul Unterlass\Anomaly_detection\01_data\{tunnel}\01_TBM_data_preprocessed.gzip')
    df = pd.read_parquet(fr'D:\02_Research\01_Unterlass\05_Anomaly_detection\01_data\{tunnel}\01_TBM_data_preprocessed.gzip')
    # df['GI'] = df['GI'] -1
    
    # hard drop of outliers which lie beyond the machine limits
    df.drop(df[df['Torque cutterhead [MNm]'] > 4.5].index, inplace=True)
    df.drop(df[df['Total advance force [kN]'] < 2000].index, inplace=True)
    df.drop(df[df['Total advance force [kN]'] > 17500].index, inplace=True)
    df.drop(df[df['Penetration [mm/rot]'] < 0.1].index, inplace=True)
    df = df[df['Tunnel Distance [m]'] > 1*1000] # first km not representative
    
    # rename columns for consistency through different datasets
    df.rename(columns={'GI': 'Class'}, inplace=True)
    df.rename(columns={'Speed cutterhead [rpm]': 'Speed cutterhead for display [rpm]'},
            inplace=True)
    df.reset_index(inplace=True, drop=True)
    
if tunnel == 'FB':
    df = pd.read_parquet(fr'E:\Paul Unterlass\Anomaly_detection\01_data\{tunnel}\01_TBM_data_preprocessed_S980.gzip')
        
    # hard drop of outliers which lie beyond the machine limits
    # df.drop(df[df['Torque cutterhead [MNm]'] > 10.2].index, inplace=True)
    # df.drop(df[df['Total advance force [kN]'] > 27000].index, inplace=True)
    # df.drop(df[df['Total advance force [kN]'] < 4000].index, inplace=True)
    # df.drop(df[df['Penetration [mm/rot]'] < 0.1].index, inplace=True)
    
    df = df[df['Tunnel Distance [m]'] > 1*1000] # first km not representative
    
    # rename columns for consistency through different datasets
    df.rename(columns={'advance': 'Class'}, inplace=True)
    
    df.rename(columns={'CH Rotation [rpm]': 'Speed cutterhead for display [rpm]'},
            inplace=True)
    df.rename(columns={'CH Penetration [mm/rot]': 'Penetration [mm/rot]'},
              inplace=True)
    df.rename(columns={'CH Torque [MNm]': 'Torque cutterhead [MNm]'},
              inplace=True)
    df.rename(columns={'Thrust Force [kN]': 'Total advance force [kN]'},
              inplace=True)
    
    df.reset_index(inplace=True, drop=True)
    
elif tunnel == 'Synth_BBT':    
    df = pd.read_parquet(fr'E:\Paul Unterlass\Anomaly_detection\01_data\{tunnel}\01_TBM_data_preprocessed.gzip')
    
    # df['GI'] = df['GI'] - 1
    
    # hard drop of outliers which lie beyond the machine limits
    df.drop(df[df['Torque cutterhead [MNm]'] > 4.5].index, inplace=True)
    df.drop(df[df['Total advance force [kN]'] < 2000].index, inplace=True)
    df.drop(df[df['Total advance force [kN]'] > 17500].index, inplace=True)
    df.drop(df[df['Penetration [mm/rot]'] < 0.1].index, inplace=True)
    df = df[df['Tunnel Distance [m]'] > 1*1000] # first km not representative
    
    # rename columns for consistency through different datasets
    df.rename(columns={'GI': 'Class'}, inplace=True)
    df.rename(columns={'Speed cutterhead [rpm]': 'Speed cutterhead for display [rpm]'},
            inplace=True)

    df.reset_index(inplace=True, drop=True)    

elif tunnel == 'Synth_BBT_UT':    
    df = pd.read_parquet(
        fr'E:\Paul Unterlass\Anomaly_detection\01_data\{tunnel}\01_TBM_data_preprocessed_Qclass.gzip')
    
    df['Class'] = df['Class'] - 1
    
    # hard drop of outliers which lie beyond the machine limits
    df.drop(df[df['Torque cutterhead [MNm]'] > 10.2].index, inplace=True)
    df.drop(df[df['Total advance force [kN]'] > 27000].index, inplace=True)
    df.drop(df[df['Total advance force [kN]'] < 4000].index, inplace=True)
    df.drop(df[df['Penetration [mm/rot]'] < 0.1].index, inplace=True)
    df.reset_index(inplace=True, drop=True)

elif tunnel == 'Synth_UT_BBT':    
    df = pd.read_parquet(
        fr'E:\Paul Unterlass\Anomaly_detection\01_data\{tunnel}\01_TBM_data_preprocessed.gzip')
        
    # hard drop of outliers which lie beyond the machine limits
    df.drop(df[df['Torque cutterhead [MNm]'] > 4.5].index, inplace=True)
    df.drop(df[df['Total advance force [kN]'] < 2000].index, inplace=True)
    df.drop(df[df['Total advance force [kN]'] > 17500].index, inplace=True)
    df.drop(df[df['Penetration [mm/rot]'] < 0.1].index, inplace=True)
    df = df[df['Tunnel Distance [m]'] > 1*1000] # first km not representative
    
    # rename columns for consistency through different datasets
    df.rename(columns={'GI': 'Class'}, inplace=True)
    df.rename(columns={'Speed cutterhead [rpm]': 'Speed cutterhead for display [rpm]'},
            inplace=True)
    df.reset_index(inplace=True, drop=True)
    
else:
    print('dataset not available')

# =============================================================================
# Define train/validation/test sections, variables and sequence length
# =============================================================================

if tunnel == 'Synth_BBT':
    columns = ['Penetration [mm/rot]',
               'Speed cutterhead for display [rpm]',
               'Torque cutterhead [MNm]',
               'Total advance force [kN]',
               'torque ratio', 
               ]
    
    target_column = ['Class']
    
    val_start = 0
    val_end = 0.25
    test_start1 = 3.5
    test_end1 = 4.5
    test_start2 = 5.0
    test_end2 = 6.0
    test_start3 = 9.5
    test_end3 = 10.5
        
    # parameters for sequences
    seq_size = 100
    
elif tunnel == 'BBT':
    
    columns = ['Advance speed [mm/min]',
               'Pressure advance cylinder bottom side [bar]',
               'Penetration [mm/rot]',
               'Speed cutterhead for display [rpm]',
               'Torque cutterhead [MNm]',
               'Total advance force [kN]',
               'spec. penetration [mm/rot/MN]', 
               'torque ratio', 
               ]
    
    target_column = ['Class']
    
    val_start = 10.5
    val_end = 11.5
    test_start1 = 3.5
    test_end1 = 4.5
    test_start2 = 5.0
    test_end2 = 6.0
    test_start3 = 9.5
    test_end3 = 10.5
        
    # parameters for sequences
    seq_size = 100
        
elif tunnel == 'UT':
    columns = ['Advance speed [mm/min]',
               'Pressure advance cylinder bottom side [bar]',
               'Penetration [mm/rot]',
               'Speed cutterhead for display [rpm]',
               'Torque cutterhead [MNm]',
               'Total advance force [kN]',
               'spec. penetration [mm/rot/MN]', 
               'torque ratio', 
               ]
    
    df_spec_tor = df[['spec. penetration [mm/rot/MN]', 
                 'torque ratio', 
                 'Tunnel Distance [m]',
                 'Class',
                 ]]
    
    target_column = ['Class']

    # df_spec_tor.to_csv('spec. penetration and torque ratio.csv', sep=',',
    #                    index=False, encoding='utf-8')
    # columns_2plot = columns + target_column
    # subplots(df, columns_2plot, 2600, 2800)

    val_start = 3.5
    val_end = 4.5
    test_start1 = 2.4 #4.9
    test_end1 = 2.8 #5.25
    test_start2 = 4.9
    test_end2 = 5.25
    test_start3 = 5.9
    test_end3 = 6.2
        
    # parameters for sequences
    seq_size = 100
    
elif tunnel == 'FB':
    
    columns = ['Penetration [mm/rot]',
               'Speed cutterhead for display [rpm]',
               'Torque cutterhead [MNm]',
               'Total advance force [kN]',
               'torque ratio', 
               ]
    
    target_column = ['Class']
    
    val_start = 0
    val_end = 0.25
    test_start1 = 3.5
    test_end1 = 4.5
    test_start2 = 5.5
    test_end2 = 6.5
    test_start3 = 7.5
    test_end3 = 8.5
        
    # parameters for sequences
    seq_size = 100

elif tunnel == 'Synth_BBT_UT':
    columns = ['Penetration [mm/rot]',
               'Speed cutterhead for display [rpm]',
               'Torque cutterhead [MNm]',
               'Total advance force [kN]',
               'torque ratio', 
               ]
    
    target_column = ['Class']
    
    val_start = 0
    val_end = 0.25
    test_start1 = 2.4 #4.9
    test_end1 = 2.8 #5.25
    test_start2 = 4.9
    test_end2 = 5.25
    test_start3 = 5.9
    test_end3 = 6.2
        
    # parameters for sequences
    seq_size = 100
    
elif tunnel == 'Synth_UT_BBT':
    columns = ['Penetration [mm/rot]',
               'Speed cutterhead for display [rpm]',
               'Torque cutterhead [MNm]',
               'Total advance force [kN]',
               'torque ratio', 
               ]
    
    target_column = ['Class']
    
    val_start = 0
    val_end = 0.25
    test_start1 = 3.5
    test_end1 = 4.5
    test_start2 = 5.0
    test_end2 = 6.0
    test_start3 = 9.5
    test_end3 = 10.5
        
    # parameters for sequences
    seq_size = 100
    
else:
    print('tunnel not defined')

# =============================================================================
# Function taht splits pre-processes train, validation and test datasets
# =============================================================================

def create_sections(tunnel,
                    dataset,
                    columns,
                    target_column,
                    test_start1,
                    test_end1,
                    test_start2,
                    test_end2,
                    test_start3,
                    test_end3,
                    validation_start,
                    validation_end):

    # creating section for test dataset
    dataset = dataset[dataset['Tunnel Distance [m]'] >= 1000]
    test_df1 = dataset[(dataset['Tunnel Distance [m]'] >= test_start1*1000) & 
                      (dataset['Tunnel Distance [m]'] < test_end1*1000)]
    test_df1 = test_df1[columns + target_column]      
    test_df1.reset_index(inplace=True, drop=True)
    
    test_df2 = dataset[(dataset['Tunnel Distance [m]'] >= test_start2*1000) & 
                  (dataset['Tunnel Distance [m]'] < test_end2*1000)]
    test_df2 = test_df2[columns + target_column]      
    test_df2.reset_index(inplace=True, drop=True)
    
    test_df3 = dataset[(dataset['Tunnel Distance [m]'] >= test_start3*1000) & 
                      (dataset['Tunnel Distance [m]'] < test_end3*1000)]
    test_df3 = test_df3[columns + target_column]      
    test_df3.reset_index(inplace=True, drop=True)
    
    
    # creating section for training dataset by taking everthing that is left from the dataset
    if tunnel == 'Synth_BBT':
        # read synth training data
        train_df = pd.read_excel('01_data/Synth_BBT/TBM_A_2_synthetic_advance.xlsx')
        df_strokes = pd.read_excel('01_data/Synth_BBT/TBM_A_2_synthetic_strokes.xlsx')
        # set artificial class column
        train_df['Class'] = 0
        # set class to 4 where irregular advance
        merged = train_df.merge(df_strokes, on='Stroke number [-]')
        merged.loc[merged['advance class mean'] == 1, 'Class'] = 5
        
        train_df = merged[train_df.columns]
        # create a copy of train_df to work with an independent DataFrame 
        train_df = train_df.copy()
        train_df.rename(columns={
            'tunnellength [m]': 'Tunnel Distance [m]'}, inplace=True)
        train_df.rename(columns={
            'rotations [rpm]': 'Speed cutterhead for display [rpm]'},
            inplace=True)
        train_df.rename(columns={
            'torque ratio [-]': 'torque ratio'}, inplace=True)

        # creating section for validation dataset
        val_df = train_df[(train_df['Tunnel Distance [m]'] >= validation_start*1000) & 
                         (train_df['Tunnel Distance [m]'] < validation_end*1000)]
        
        # drop val from train df
        train_df = train_df.drop(val_df.index)
        train_df.reset_index(inplace=True, drop=True)
        val_df.reset_index(inplace=True, drop=True)
        
    elif tunnel == 'Synth_BBT_UT':
        # read synth training data
        train_df = pd.read_excel('01_data/Synth_BBT/TBM_A_2_synthetic_advance.xlsx')
        df_strokes = pd.read_excel('01_data/Synth_BBT/TBM_A_2_synthetic_strokes.xlsx')
        # set artificial class column
        train_df['Class'] = 0
        # set class to 4 where irregular advance
        merged = train_df.merge(df_strokes, on='Stroke number [-]')
        merged.loc[merged['advance class mean'] == 1, 'Class'] = 4
        
        train_df = merged[train_df.columns]
        # create a copy of train_df to work with an independent DataFrame 
        train_df = train_df.copy()
        train_df.rename(columns={
            'tunnellength [m]': 'Tunnel Distance [m]'}, inplace=True)
        train_df.rename(columns={
            'rotations [rpm]': 'Speed cutterhead for display [rpm]'},
            inplace=True)
        train_df.rename(columns={
            'torque ratio [-]': 'torque ratio'}, inplace=True)

        # creating section for validation dataset
        val_df = train_df[(train_df['Tunnel Distance [m]'] >= validation_start*1000) & 
                         (train_df['Tunnel Distance [m]'] < validation_end*1000)]
        
        # drop val from train df
        train_df = train_df.drop(val_df.index)
        train_df.reset_index(inplace=True, drop=True)
        val_df.reset_index(inplace=True, drop=True)

    else:
        # creating section for validation dataset
        val_df = dataset[(dataset['Tunnel Distance [m]'] >= validation_start*1000) & 
                         (dataset['Tunnel Distance [m]'] < validation_end*1000)]
        val_df = val_df[columns + target_column]      
        val_df = val_df.reset_index()  
        
        train_df = pd.concat([
            dataset[dataset['Tunnel Distance [m]'] < test_start1*1000],
            dataset[(dataset['Tunnel Distance [m]'] >= test_end1*1000) & (dataset['Tunnel Distance [m]'] < test_start2*1000)],
            dataset[(dataset['Tunnel Distance [m]'] >= validation_end*1000) & (dataset['Tunnel Distance [m]'] < validation_start*1000)],
            dataset[(dataset['Tunnel Distance [m]'] >= test_end2*1000) & (dataset['Tunnel Distance [m]'] < test_start3*1000)],
            dataset[dataset['Tunnel Distance [m]'] >= test_start3*1000]])
        
        train_df = train_df[columns + target_column]
        train_df.reset_index(inplace=True)
    
    # ========================================================================
    # count and print class distribution before removing bad rock mass
    # conditions from training data
    # ========================================================================

    print(f'before removing classes >= {Class}\n')
    
    for i in range(6):
        count = (train_df['Class'] == i).sum()
        print(f'total occurences of Class{i} in train dataset: {count}')
        
    for i in range(6):
        count = (val_df['Class'] == i).sum()
        print(f'total occurences of Class{i} in validation dataset: {count}')
        
    for i in range(6):
        count = (test_df1['Class'] == i).sum()
        print(f'total occurences of Class{i} in test dataset1: {count}')
    for i in range(6):
        count = (test_df2['Class'] == i).sum()
        print(f'total occurences of Class{i} in test dataset2: {count}')
    for i in range(6):
        count = (test_df3['Class'] == i).sum()
        print(f'total occurences of Class{i} in test dataset3: {count}')


    # ========================================================================
    # Remvoing bad rock mass conditions from training/validation data to get
    # anomaly free train/val sets, linear interpolation between observations
    # neighbouring the removed ones
    # ========================================================================
    
    for df in [val_df, train_df]:
    
        for col in columns:
            
            # Find indices where rock mass condition >= Class
            indices = df.index[df['Class'] >= Class].tolist()
            
            # Loop through each index and perform linear interpolation
            # 5 indices before and after removed one
            
            for idx in indices:
                before_idx = idx - 5
                after_idx = idx + 5
        
                while df.at[before_idx, 'Class'] >= Class:
                    before_idx -= 5
        
                while df.at[after_idx, 'Class'] >= Class:
                    after_idx += 5
        
                # Linear interpolation overwrites Class => set Class value
                df.at[idx, col] = np.interp(idx,
                                            [before_idx, after_idx],
                                            [df.at[before_idx, col],
                                             df.at[after_idx, col]])
                df.at[idx, 'Class'] = 99

    # ========================================================================
    # count and print class distribution after removing bad rock mass
    # conditions from training data
    # ========================================================================
    print(f'\nafter removing classes >= {Class}\n')
    
    for i in range(6):
        count = (train_df['Class'] == i).sum()
        print(f'total occurences of Class{i} in train dataset: {count}')
        
    for i in range(6):
        count = (val_df['Class'] == i).sum()
        print(f'total occurences of Class{i} in validation dataset: {count}')
    
    # ========================================================================
    # define and apply scaler
    # ========================================================================
    test_target_df1 = test_df1[target_column]
    test_target_df2 = test_df2[target_column]
    test_target_df3 = test_df3[target_column]
    val_target_df = val_df[target_column]
    train_target_df = train_df[target_column]
    
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler = scaler.fit(train_df[columns])
    
    # applies scalar 
    test_df1 = pd.DataFrame(scaler.transform(test_df1[columns]), 
                                index=test_df1[columns].index, 
                                columns=[columns]
                                )
    
    test_df2 = pd.DataFrame(scaler.transform(test_df2[columns]), 
                                index=test_df2[columns].index, 
                                columns=[columns]
                                )
    
    test_df3 = pd.DataFrame(scaler.transform(test_df3[columns]), 
                                index=test_df3[columns].index, 
                                columns=[columns]
                                )
    
    val_df = pd.DataFrame(scaler.transform(val_df[columns]), 
                                index=val_df[columns].index, 
                                columns=[columns]
                                )
    
    train_df = pd.DataFrame(scaler.transform(train_df[columns]), 
                                index=train_df[columns].index, 
                                columns=[columns]
                                )

    return test_df1, test_df2, test_df3, val_df, train_df, test_target_df1, test_target_df2, test_target_df3, val_target_df, train_target_df

# =============================================================================
# Function that creates sequences for LSTM architecture
# =============================================================================

def create_sequences(dataset, target_df, seq_size):  
    
    sequences = []
    
    for i in range(len(dataset) - seq_size):
        
        sequence = dataset[i:i+seq_size]
        label_pos = i + seq_size - 1
        label = target_df.iloc[label_pos]
        sequences.append((sequence, label))
        
    return sequences

# =============================================================================
# Function that creates torch datasets
# =============================================================================

class TBMDataset(Dataset):
    
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        sequence = torch.tensor(sequence.to_numpy())
        if sequence.shape[1] != 8:
            raise RuntimeError(f"Expected input size {8}, got {sequence.shape}")
        label = torch.tensor(label)
        
        return sequence, label

# =============================================================================
# Exectute pre-processing functions
# =============================================================================
if __name__ == "__main__":
    # create train/test/val datasets
    test_df1, test_df2, test_df3, val_df, train_df, test_target_df1, test_target_df2, test_target_df3, val_target_df, train_target_df = create_sections(
        tunnel,
        df,
        columns,
        target_column,
        test_start1,
        test_end1,
        test_start2,
        test_end2,
        test_start3,
        test_end3,
        val_start,
        val_end
        )   
    
    # create sequences
    test_df1 = create_sequences(test_df1, test_target_df1, seq_size)
    test_df2 = create_sequences(test_df2, test_target_df2, seq_size)
    test_df3 = create_sequences(test_df3, test_target_df3, seq_size)
    val_df = create_sequences(val_df, val_target_df, seq_size)
    train_df = create_sequences(train_df, train_target_df, seq_size)
    
    # create torch datasets
    train_df = TBMDataset(train_df)
    test_df1 = TBMDataset(test_df1)
    test_df2 = TBMDataset(test_df2)
    test_df3 = TBMDataset(test_df3)
    val_df = TBMDataset(val_df)
    
    # define strings for filenames when saving
    test_str1 = f'{tunnel}_{Class}_test_dataset1_seq{seq_size}_{test_start1}-{test_end1}.pt'
    test_str2 = f'{tunnel}_{Class}_test_dataset2_seq{seq_size}_{test_start2}-{test_end2}.pt'
    test_str3 = f'{tunnel}_{Class}_test_dataset3_seq{seq_size}_{test_start3}-{test_end3}.pt'
    val_str = f'{tunnel}_{Class}_val_dataset_seq{seq_size}_{val_start}-{val_end}.pt'
    train_str = f'{tunnel}_{Class}_train_dataset_seq{seq_size}.pt'
    
    # save torch datasets
    torch.save(test_df1, f'01_data/{tunnel}/datasets/' + test_str1)
    torch.save(test_df2, f'01_data/{tunnel}/datasets/' + test_str2)
    torch.save(test_df3, f'01_data/{tunnel}/datasets/' + test_str3)
    torch.save(val_df, f'01_data/{tunnel}/datasets/' + val_str)
    torch.save(train_df, f'01_data/{tunnel}/datasets/' + train_str)
    
    print(f'Trainingdataset saved as {train_str}',
          f'\nTestdataset1 saved as {test_str1}',
          f'\nTestdataset2 saved as {test_str2}',
          f'\nTestdataset3 saved as {test_str3}',
          f'\nValidationdataset saved as {val_str}',
          '\n\npre-processing finished')
    
