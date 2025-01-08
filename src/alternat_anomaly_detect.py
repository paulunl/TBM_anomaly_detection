# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 13:28:12 2024

@author: Rechenknecht
"""

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 15)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters
from mpl_toolkits.mplot3d import Axes3D

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
register_matplotlib_converters()
from time import time
import seaborn as sns
sns.set(style="whitegrid")

# =============================================================================
# load data
# =============================================================================

tunnel = 'FB' # 'BBT' # 'Synth_BBT_UT'
Class = 2

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


columns = ['Tunnel Distance [m]',
           'Advance speed [mm/min]',
           'Pressure advance cylinder bottom side [bar]',
           'Penetration [mm/rot]',
           'Speed cutterhead for display [rpm]',
           'Torque cutterhead [MNm]',
           'Total advance force [kN]',
           'spec. penetration [mm/rot/MN]', 
           'torque ratio', 
           ]

rmc = ['Class']

df = df[columns + rmc]      

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
    
df1 = df[(df['Tunnel Distance [m]'] >= test_start1*1000) & 
                  (df['Tunnel Distance [m]'] < test_end1*1000)]

df2 = df[(df['Tunnel Distance [m]'] >= test_start2*1000) & 
                  (df['Tunnel Distance [m]'] < test_end2*1000)]

df3 = df[(df['Tunnel Distance [m]'] >= test_start3*1000) & 
                  (df['Tunnel Distance [m]'] < test_end3*1000)]
# =============================================================================
# Isolation Forest
# =============================================================================

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

def isolation_forest(df):
    outliers_fraction = float(.02)
    
    # scale
    scaler = StandardScaler()
    
    np_scaled = scaler.fit_transform(df[columns].values)
    data = pd.DataFrame(np_scaled, columns = columns)
    
    # train isolation forest
    model = IsolationForest(contamination=outliers_fraction)
    model.fit(data) 
    
    df['anomaly'] = model.predict(data)
    
    return df

df1 = isolation_forest(df1)
df2 = isolation_forest(df2)
df3 = isolation_forest(df3)

def visualize(df, test_start, test_end):
    # visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10,6))
    
    a = df.loc[df['anomaly'] == -1] #anomaly
    
    # ax1.plot(df['Tunnel Distance [m]'], df['Penetration [mm/rot]'].rolling(35).mean(),
    #          color='black', label='Normal')
    # ax1.plot(df['Tunnel Distance [m]'], df['Penetration [mm/rot]'], color='black',
    #          alpha=0.35)
    # ax1.scatter(a['Tunnel Distance [m]'], a['Penetration [mm/rot]'], color='red',
    #             label='Anomaly')
    # ax1.set_ylabel('Penetration [mm/rot]')
    ax1.plot(df['Tunnel Distance [m]'], df['Total advance force [kN]'].rolling(35).mean(),
             color='black', label='Normal')
    ax1.plot(df['Tunnel Distance [m]'], df['Total advance force [kN]'], color='black',
             alpha=0.35)
    ax1.scatter(a['Tunnel Distance [m]'], a['Total advance force [kN]'], color='red',
                label='Anomaly')
    ax1.set_ylabel('Total advance force [kN]')
    ax1.set_xlim(test_start*1000, test_end*1000)
    ax1.legend(loc='lower left')
    
    ax2.plot(df['Tunnel Distance [m]'], df['spec. penetration [mm/rot/MN]'].rolling(35).mean(),
             color='black')
    ax2.plot(df['Tunnel Distance [m]'], df['spec. penetration [mm/rot/MN]'],
             color='black', alpha=0.35)
    ax2.scatter(a['Tunnel Distance [m]'], a['spec. penetration [mm/rot/MN]'],
                color='red')
    ax2.set_ylabel('spec. penetration [mm/rot/MN]')
    ax2.set_xlim(test_start*1000, test_end*1000)
    
    ax3.plot(df['Tunnel Distance [m]'], df['torque ratio'].rolling(35).mean(),
             color='black')
    ax3.plot(df['Tunnel Distance [m]'], df['torque ratio'], color='black',
             alpha=0.35)
    ax3.scatter(a['Tunnel Distance [m]'], a['torque ratio'], color='red')
    ax3.set_ylabel('toque ratio')
    ax3.set_xlabel('Tunnel Distance [m]')
    ax3.set_xlim(test_start*1000, test_end*1000)
    
    n_clusters = df['Class'].max() - df['Class'].min() + 1
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
            
    elif n_clusters == 4 and df['Class'].min() == 0:
        cmap = (mpl.colors.ListedColormap(['green',
                                       'greenyellow',
                                       'gold',
                                       'orange'])
                .with_extremes(over='red', under='green'))
    
    elif n_clusters == 4 and df['Class'].min() == 1:
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
    
    
    c = ax1.pcolorfast(ax1.get_xlim(),
                      ax1.get_ylim(),
                      df['Class'].values[np.newaxis],
                      cmap=cmap,
                      vmin=(df['Class'].min()),
                      vmax=(df['Class'].max()+1),
                      alpha=0.5)
    
    cbar = fig.colorbar(c, ax=ax1) # , spacing='uniform'
    
    c = ax2.pcolorfast(ax2.get_xlim(),
                      ax2.get_ylim(),
                      df['Class'].values[np.newaxis],
                      cmap=cmap,
                      vmin=(df['Class'].min()),
                      vmax=(df['Class'].max()+1),
                      alpha=0.5)
    
    cbar = fig.colorbar(c, ax=ax2) # , spacing='uniform'
    
    c = ax3.pcolorfast(ax3.get_xlim(),
                      ax3.get_ylim(),
                      df['Class'].values[np.newaxis],
                      cmap=cmap,
                      vmin=(df['Class'].min()),
                      vmax=(df['Class'].max()+1),
                      alpha=0.5)
    
    cbar = fig.colorbar(c, ax=ax3) # , spacing='uniform'
    
    
    tick_locs = np.linspace(int(df['Class'].min()),
                            int(df['Class'].max()+1),
                            int(2 * n_clusters + 1))[1::2]
    
    tick_label = np.arange(int(df['Class'].min()),
                                int(df['Class'].max() + 1))
    
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(tick_label)
    
    cbar.ax.set_ylim(df['Class'].min(), int(df['Class'].max()+1))
    
    fig.tight_layout()
    
visualize(df1, test_start1, test_end1)
visualize(df2, test_start2, test_end2)
visualize(df3, test_start3, test_end3)



