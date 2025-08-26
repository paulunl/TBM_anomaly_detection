# -*- coding: utf-8 -*-
"""
TBM Operational Data-Driven Anomaly Detection in Hard Rock Excavations

Script containing helper functions:
- **concat**: Reads and merges multiple CSV files of raw TBM data into a single dataset.
- **preprocessor**: Applies a preprocessing routine to TBM data.

@author: Paul Unterlaß
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from VAE_00_utilities import utilities, computation  # Import utility classes

def concat():
    """
    Reads multiple CSV files from a specified folder and concatenates them into a single dataset.
    Can optionally remove standstill records and check for missing values.
    """
    folder = r'...'  # Set the folder containing CSV files
    drop_standstills = True  # Drop standstill records if True
    check_for_miss_vals = False  # Generate missing value plot if True

    def concat_tables(folder, drop_standstills=False, check_for_miss_vals=True):
        """
        Reads CSV files from a folder and merges them into a DataFrame.
        """
        filenames = [file for file in os.listdir(folder) if file.endswith('.csv')]
        files = []

        for i, file in enumerate(filenames[:1000]):  # Limit processing to 1000 files
            df = pd.read_csv(os.path.join(folder, file), sep=';')
            print(f'Processing {file} ({i+1}/{len(filenames)})')
            
            df.drop(columns=df.columns[0], inplace=True)
            df.columns = [
                'Date', 'Stroke', 'Chainage [m]', 'Tunnel Distance [m]',
                'Advance speed [mm/min]', 'Speed cutterhead [rpm]',
                'Pressure advance cylinder bottom side [bar]',
                'Torque cutterhead [MNm]', 'Total advance force [kN]',
                'Penetration [mm/rot]', 'Pressure RSC left [bar]',
                'Pressure RSC right [bar]', 'Path RSC left [mm]'
            ]
            
            df = df[df['Tunnel Distance [m]'].ge(0)]  # Remove negative tunnel distances
            files.append(df)

        df_combined = pd.concat(files, sort=True).dropna()
        if check_for_miss_vals:
            utilities().check_for_miss_vals(df_combined)
        
        return df_combined
    
    df = concat_tables(folder, drop_standstills, check_for_miss_vals)
    output_path = r'03_data\01_TBMdata.gzip' if drop_standstills else r'03_data\00_TBMdata_wStandstills.gzip'
    df.to_parquet(output_path, index=False)
    return df

def preprocessor():
    """
    Preprocessing routine for TBM operational data.
    - Reads raw data
    - Removes outliers
    - Applies linear interpolation
    - Computes additional features
    - Saves processed data
    """
    utils = utilities()
    comp = computation()

    # Load concatenated TBM data
    df = pd.read_parquet(r'03_data\01_TBMdata.gzip')
    df = df[['Date', 'Stroke', 'Chainage [m]', 'Tunnel Distance [m]',
             'Advance speed [mm/min]', 'Speed cutterhead [rpm]',
             'Pressure advance cylinder bottom side [bar]',
             'Torque cutterhead [MNm]', 'Total advance force [kN]',
             'Penetration [mm/rot]']]
    
    print(f'Initial datapoints (without standstills): {df.shape[0]}')

    # Group by tunnel distance and take median
    df_median = df.groupby('Tunnel Distance [m]', as_index=False).median()
    print(f'Datapoints after grouping: {df_median.shape[0]}')

    # Outlier filtering using Mahalanobis distance
    mahal_features = [
        'Advance speed [mm/min]', 'Speed cutterhead [rpm]',
        'Pressure advance cylinder bottom side [bar]',
        'Torque cutterhead [MNm]', 'Total advance force [kN]',
        'Penetration [mm/rot]'
    ]
    df_filtered = utils.filter_outliers(df_median, mahal_features, 85, 100)
    df_filtered.sort_values(by=['Tunnel Distance [m]'], inplace=True)
    df_filtered.reset_index(drop=True, inplace=True)
    print(f'Datapoints after outlier removal: {df_filtered.shape[0]}')

    # Linear interpolation for equal spacing
    interval = round(df_filtered['Tunnel Distance [m]'].diff().median(), 2)
    df_equal = utils.equally_spaced_df(df_filtered, 'Tunnel Distance [m]', interval)
    print(f'Datapoints after interpolation: {df_equal.shape[0]}')

    # Feature engineering
    df = df_equal
    df['spec. penetration [mm/rot/MN]'] = comp.s_pen(df['Penetration [mm/rot]'], df['Total advance force [kN]'])
    
    # Compute torque ratio
    tot_cutters = 46
    r_cutter = 241.3  # [mm]
    M0 = df['Torque cutterhead [MNm]'].mean() * 1000
    real_torque = df['Torque cutterhead [MNm]'] * 1000
    cutter_positions = 0.3 * (7.93 / 2)
    df['torque ratio'], df['theoretical torque [MNm]'] = comp.t_ratio(
        tot_cutters, r_cutter, M0, df['Total advance force [kN]'],
        df['Penetration [mm/rot]'], real_torque, cutter_positions
    )
    
    # Final formatting and saving
    df = df.round({'Tunnel Distance [m]': 2}).dropna()
    df.to_parquet(r'03_data\02_TBMdata_preprocessed.gzip', index=False)
    return df

if __name__ == "__main__":
    df = concat()
    df = preprocessor()
