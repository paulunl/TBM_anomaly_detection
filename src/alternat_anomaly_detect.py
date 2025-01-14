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
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# =============================================================================
# load data
# =============================================================================
def load_data(tunnel, Class):    
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
        # df = pd.read_parquet(fr'E:\Paul Unterlass\Anomaly_detection\01_data\{tunnel}\01_TBM_data_preprocessed_S980.gzip')
        df = pd.read_parquet(fr'D:\02_Research\01_Unterlass\05_Anomaly_detection\01_data\{tunnel}\01_TBM_data_preprocessed_S980.gzip')
    
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
        df.rename(columns={'Speed [mm/min]': 'Advance speed [mm/min]'},
                  inplace=True)
        
        df.reset_index(inplace=True, drop=True)
    
    
    columns = ['Tunnel Distance [m]',
               'Advance speed [mm/min]',
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
        
    if tunnel == 'BBT':
        
        columns = ['Advance speed [mm/min]',
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
        
        columns = ['Advance speed [mm/min]',
                   'Penetration [mm/rot]',
                   'Speed cutterhead for display [rpm]',
                   'Torque cutterhead [MNm]',
                   'Total advance force [kN]',
                   'spec. penetration [mm/rot/MN]', 
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
        
    df1 = df[(df['Tunnel Distance [m]'] >= test_start1*1000) & 
                      (df['Tunnel Distance [m]'] < test_end1*1000)]
    
    df2 = df[(df['Tunnel Distance [m]'] >= test_start2*1000) & 
                      (df['Tunnel Distance [m]'] < test_end2*1000)]
    
    df3 = df[(df['Tunnel Distance [m]'] >= test_start3*1000) & 
                      (df['Tunnel Distance [m]'] < test_end3*1000)]
    
    return test_start1, test_start2, test_start3, test_end1, test_end2, test_end3, columns, df, df1, df2, df3

tunnel = 'BBT' # 'BBT'
Class = 3 #3 BBT #4 UT #1 FB

test_start1, test_start2, test_start3, test_end1, test_end2, test_end3, columns, df, df1, df2, df3 = load_data(tunnel, Class)

# =============================================================================
# Isolation Forest
# =============================================================================

from sklearn.preprocessing import StandardScaler
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

def visualize(df, test_start, test_end, file_name):
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
             alpha=0.5)
    ax1.scatter(a['Tunnel Distance [m]'], a['Total advance force [kN]'], color='red',
                label='Anomaly')
    ax1.set_ylabel('Total advance force [kN]')
    ax1.set_xlim(test_start*1000, test_end*1000)
    ax1.legend(loc='best')
    
    ax2.plot(df['Tunnel Distance [m]'], df['spec. penetration [mm/rot/MN]'].rolling(35).mean(),
             color='black')
    ax2.plot(df['Tunnel Distance [m]'], df['spec. penetration [mm/rot/MN]'],
             color='black', alpha=0.5)
    ax2.scatter(a['Tunnel Distance [m]'], a['spec. penetration [mm/rot/MN]'],
                color='red')
    ax2.set_ylabel('spec. penetration \n[mm/rot/MN]')
    ax2.set_xlim(test_start*1000, test_end*1000)
    
    ax3.plot(df['Tunnel Distance [m]'], df['torque ratio'].rolling(35).mean(),
             color='black')
    ax3.plot(df['Tunnel Distance [m]'], df['torque ratio'], color='black',
             alpha=0.5)
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
                                       'red']))
    elif n_clusters == 5:
        cmap = (mpl.colors.ListedColormap(['green',
                                       'greenyellow',
                                       'gold',
                                       'orange',
                                       'darkorange']))
    elif n_clusters == 4 and tunnel == 'Synth_BBT':
        cmap = (mpl.colors.ListedColormap([
                                       'green',
                                       'gold',
                                       'orange',
                                       'red']))
    elif n_clusters == 4 and tunnel == 'BBT':
        cmap = (mpl.colors.ListedColormap([
                                       'green',
                                       'gold',
                                       'orange',
                                       'red']))    
    # elif n_clusters == 4 and df['Label'].min() == 0:
    #     cmap = (mpl.colors.ListedColormap(['green',
    #                                    'greenyellow',
    #                                    'gold',
    #                                    'orange'])
    #             .with_extremes(over='red', under='green'))
    elif n_clusters == 4 and df['Class'].min() == 1:
        cmap = (mpl.colors.ListedColormap([
                                       'greenyellow',
                                       'gold',
                                       'orange',
                                       'darkorange']))
    elif n_clusters == 3:
        cmap = (mpl.colors.ListedColormap(['green',
                                       'gold',
                                       'red']))
        
    elif n_clusters == 2:
        cmap = (mpl.colors.ListedColormap(['green',
                                       'gold']))
        
    else:
        print('colour map not suited for the data to be plotted')    
    
    #cbar ax1
    c = ax1.pcolorfast(ax1.get_xlim(),
                      ax1.get_ylim(),
                      df['Class'].values[np.newaxis],
                      cmap=cmap,
                      vmin=(df['Class'].min()),
                      vmax=(df['Class'].max()+1),
                      alpha=0.5)
    
    cbar = fig.colorbar(c, ax=ax1) # , spacing='uniform'
    
    tick_locs = np.linspace(int(df['Class'].min()),
                            int(df['Class'].max()+1),
                            int(2 * n_clusters + 1))[1::2]
    
    tick_label = np.arange(int(df['Class'].min()),
                                int(df['Class'].max() + 1))
    
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(tick_label)
    
    cbar.ax.set_ylim(df['Class'].min(), int(df['Class'].max()+1))
    
    # cbar ax2
    c = ax2.pcolorfast(ax2.get_xlim(),
                      ax2.get_ylim(),
                      df['Class'].values[np.newaxis],
                      cmap=cmap,
                      vmin=(df['Class'].min()),
                      vmax=(df['Class'].max()+1),
                      alpha=0.5)
    
    cbar = fig.colorbar(c, ax=ax2) # , spacing='uniform'
    
    tick_locs = np.linspace(int(df['Class'].min()),
                            int(df['Class'].max()+1),
                            int(2 * n_clusters + 1))[1::2]
    
    tick_label = np.arange(int(df['Class'].min()),
                                int(df['Class'].max() + 1))
    
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(tick_label)
    
    cbar.ax.set_ylim(df['Class'].min(), int(df['Class'].max()+1))
    
    # cbar ax3
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
    plt.savefig(fr'02_Results\{tunnel}\02_Plots\04_isolation_forest\{file_name}_{tunnel}.png', dpi=300)

visualize(df1, test_start1, test_end1, 'test_set1')
visualize(df2, test_start2, test_end2, 'test_set2')
visualize(df3, test_start3, test_end3, 'test_set3')

# =============================================================================
# Forecasting using VAR (vector auto regression)
# =============================================================================

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

# Remove unecessary columns

df1_VAR = df1[columns]
df2_VAR = df2[columns]
df3_VAR = df3[columns]

# Check if data is stationary (i.e. statistical properties do not change
# over time) using the Augmented Dickey-Fuller (ADF) Test

# Function to perform ADF test
def adf_test(series):
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"   {key}: {value}")
    if result[1] < 0.05:
        print("Conclusion: The series is stationary.")
    else:
        print("Conclusion: The series is non-stationary.")

# Apply ADF test to each time series in the dataset
for column in df1_VAR.columns:
    print(f"\nADF Test for {column}:")
    adf_test(df1_VAR[column])
    
for column in df2_VAR.columns:
    print(f"\nADF Test for {column}:")
    adf_test(df2_VAR[column])
    
for column in df3_VAR.columns:
    print(f"\nADF Test for {column}:")
    adf_test(df3_VAR[column])
    
def detect_anomalies_var(data):
    """
    Detect anomalies in multivariate time series using VAR.
    
    Parameters:
    - data: pd.DataFrame, multivariate time series (columns are variables).
    - lag_order: int, lag order for the VAR model.
    - threshold: float, number of standard deviations for anomaly detection.
    
    Returns:
    - anomalies: pd.DataFrame, binary (1 = anomaly, 0 = normal) for each variable.
    - residuals: pd.DataFrame, residuals of the VAR model for each variable.
    """
    # Fit the VAR model
    model = VAR(data)
    # Find the optimal lag order
    lag_order_results = model.select_order(maxlags=15)
    print(lag_order_results.summary())
    optimal_lag = lag_order_results.aic
    results = model.fit(optimal_lag)
    
    # Get the model's predicted values
    predicted = results.fittedvalues
    
    # Compute residuals
    residuals = data[optimal_lag:] - predicted
    
    # Initialize anomaly DataFrame
    anomalies = pd.DataFrame(0, index=residuals.index, columns=data.columns)
    
    # Dynamic threshold based on percentils
    for column in residuals.columns:
        threshold = residuals[column].quantile(0.99)
        anomalies[column] = (np.abs(residuals[column]) > threshold).astype(int)
    
    return anomalies, residuals

test1_anom, test1_resid = detect_anomalies_var(df1_VAR)
test2_anom, test2_resid = detect_anomalies_var(df2_VAR)
test3_anom, test3_resid = detect_anomalies_var(df3_VAR)


df1_VAR['Tunnel Distance [m]'] = df1['Tunnel Distance [m]'] 
df1_VAR['Class'] = df1['Class']
df2_VAR['Tunnel Distance [m]'] = df2['Tunnel Distance [m]']
df2_VAR['Class'] = df2['Class']
df3_VAR['Tunnel Distance [m]'] = df3['Tunnel Distance [m]']
df3_VAR['Class'] = df3['Class']

# Plot results

def visualize_VAR(df, anomalies, test_start, test_end, file_name):
    # visualization
    df = df.iloc[15:]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10,6))
    
    a = anomalies['Total advance force [kN]'] == 1
    
    ax1.plot(df['Tunnel Distance [m]'], df['Total advance force [kN]'].rolling(35).mean(),
             color='black', label='Normal')
    ax1.plot(df['Tunnel Distance [m]'], df['Total advance force [kN]'], color='black',
             alpha=0.5)
    ax1.scatter(df.loc[a, 'Tunnel Distance [m]'],
                df.loc[a, 'Total advance force [kN]'],
                color='red',
                label='Anomaly')
    ax1.set_ylabel('Total advance force [kN]')
    ax1.set_xlim(test_start*1000, test_end*1000)
    ax1.legend(loc='best')
    
    a = anomalies['spec. penetration [mm/rot/MN]'] == 1
    
    ax2.plot(df['Tunnel Distance [m]'], df['spec. penetration [mm/rot/MN]'].rolling(35).mean(),
             color='black')
    ax2.plot(df['Tunnel Distance [m]'], df['spec. penetration [mm/rot/MN]'],
             color='black', alpha=0.5)
    ax2.scatter(df.loc[a, 'Tunnel Distance [m]'],
                df.loc[a, 'spec. penetration [mm/rot/MN]'],
                color='red',
                label='Anomaly')
    ax2.set_ylabel('spec. penetration \n[mm/rot/MN]')
    ax2.set_xlim(test_start*1000, test_end*1000)

    a = anomalies['torque ratio'] == 1
    ax3.plot(df['Tunnel Distance [m]'], df['torque ratio'].rolling(35).mean(),
             color='black')
    ax3.plot(df['Tunnel Distance [m]'], df['torque ratio'], color='black',
             alpha=0.5)
    ax3.scatter(df.loc[a, 'Tunnel Distance [m]'],
                df.loc[a, 'torque ratio'],
                color='red',
                label='Anomaly')
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
                                       'red']))
    elif n_clusters == 5:
        cmap = (mpl.colors.ListedColormap(['green',
                                       'greenyellow',
                                       'gold',
                                       'orange',
                                       'darkorange']))
    elif n_clusters == 4 and tunnel == 'Synth_BBT':
        cmap = (mpl.colors.ListedColormap([
                                       'green',
                                       'gold',
                                       'orange',
                                       'red']))
    elif n_clusters == 4 and tunnel == 'BBT':
        cmap = (mpl.colors.ListedColormap([
                                       'green',
                                       'gold',
                                       'orange',
                                       'red']))    
    # elif n_clusters == 4 and df['Label'].min() == 0:
    #     cmap = (mpl.colors.ListedColormap(['green',
    #                                    'greenyellow',
    #                                    'gold',
    #                                    'orange'])
    #             .with_extremes(over='red', under='green'))
    elif n_clusters == 4 and df['Class'].min() == 1:
        cmap = (mpl.colors.ListedColormap([
                                       'greenyellow',
                                       'gold',
                                       'orange',
                                       'darkorange']))
    elif n_clusters == 3:
        cmap = (mpl.colors.ListedColormap(['green',
                                       'gold',
                                       'red']))
        
    elif n_clusters == 2:
        cmap = (mpl.colors.ListedColormap(['green',
                                       'gold']))
        
    else:
        print('colour map not suited for the data to be plotted')    
    
    #cbar ax1
    c = ax1.pcolorfast(ax1.get_xlim(),
                      ax1.get_ylim(),
                      df['Class'].values[np.newaxis],
                      cmap=cmap,
                      vmin=(df['Class'].min()),
                      vmax=(df['Class'].max()+1),
                      alpha=0.5)
    
    cbar = fig.colorbar(c, ax=ax1) # , spacing='uniform'
    
    tick_locs = np.linspace(int(df['Class'].min()),
                            int(df['Class'].max()+1),
                            int(2 * n_clusters + 1))[1::2]
    
    tick_label = np.arange(int(df['Class'].min()),
                                int(df['Class'].max() + 1))
    
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(tick_label)
    
    cbar.ax.set_ylim(df['Class'].min(), int(df['Class'].max()+1))
    
    # cbar ax2
    c = ax2.pcolorfast(ax2.get_xlim(),
                      ax2.get_ylim(),
                      df['Class'].values[np.newaxis],
                      cmap=cmap,
                      vmin=(df['Class'].min()),
                      vmax=(df['Class'].max()+1),
                      alpha=0.5)
    
    cbar = fig.colorbar(c, ax=ax2) # , spacing='uniform'
    
    tick_locs = np.linspace(int(df['Class'].min()),
                            int(df['Class'].max()+1),
                            int(2 * n_clusters + 1))[1::2]
    
    tick_label = np.arange(int(df['Class'].min()),
                                int(df['Class'].max() + 1))
    
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(tick_label)
    
    cbar.ax.set_ylim(df['Class'].min(), int(df['Class'].max()+1))
    
    # cbar ax3
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
    plt.savefig(fr'02_Results\{tunnel}\02_Plots\05_forecasting_VAR\{file_name}_{tunnel}.png', dpi=300)

visualize_VAR(df1_VAR, test1_anom, test_start1, test_end1, 'test_set1')
visualize_VAR(df2_VAR, test2_anom, test_start2, test_end2, 'test_set2')
visualize_VAR(df3_VAR, test3_anom, test_start3, test_end3, 'test_set3')

# =============================================================================
# Clustering-based anomaly detection
# =============================================================================
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min

# load test datasets, drop Class and Tunnel Distance
test_start1, test_start2, test_start3, test_end1, test_end2, test_end3, columns, df, df1, df2, df3 = load_data(tunnel, Class)

df1 = df1.drop(columns=['Class', 'Tunnel Distance [m]'])
df2 = df2.drop(columns=['Class', 'Tunnel Distance [m]'])
df3 = df3.drop(columns=['Class', 'Tunnel Distance [m]'])

# KMEANS
## Elbow Method
dfs = {
    "test_set1": df1,
    "test_set2": df2,
    "test_set3": df3}

for test_set, df in dfs.items():
    
    # Scale the data using standardscaler
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    
    n_clusters = range(1, 11)
    
    # calculate inertia for different cluster numbers
    inertia = []
    for i in n_clusters:
        model = KMeans(n_clusters=i, random_state=42)
        model.fit(df)
        inertia.append(model.inertia_)
    
    # Find the optimal number of clusters using KneeLocator
    knee_locator = KneeLocator(n_clusters,
                               inertia,
                               curve="convex",
                               direction="decreasing")
    optimal_clusters = knee_locator.knee

    # Plot the elbow curve
    # plt.plot(n_clusters, inertia, marker='o')
    # plt.axvline(x=optimal_clusters, color='r', linestyle='--',
    #             label=f'{test_set} Optimal Clusters: {optimal_clusters}')
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Inertia')
    # plt.title('Elbow Curve')
    # plt.legend()
    print(f"The optimal number of clusters in {test_set} is: {optimal_clusters}")
    
    # determine number of features to keep
    # Fit PCA to data
    pca = PCA()
    pca.fit(df)
    
    # Calculate the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    
    # Calculate the cumulative explained variance
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    
    # Plot the explained variance and cumulative explained variance
    # plt.figure(figsize=(8, 6))
    # plt.plot(range(1, len(explained_variance_ratio) + 1),
    #          cumulative_explained_variance, marker='o', color='b')
    # plt.axhline(y=0.90, color='r', linestyle='--')  # 90% threshold
    # plt.xlabel('Number of Principal Components')
    # plt.ylabel('Cumulative Explained Variance')
    # plt.title('Explained Variance vs. Number of Components')
    # plt.grid(True)
    # plt.show()
    
    # Automatically choose the number of components for 90% explained variance
    n_components = np.argmax(cumulative_explained_variance >= 0.95) + 1
    print(f"Number of components to keep: {n_components}")
    
    # Apply PCA with the selected number of components
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(df)
    
    # Determine Anomalys
    # Fit the KMeans model
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    kmeans.fit(reduced_data)
    
    # Get cluster labels and centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    # Calculate distance from each point to its nearest centroid
    distances = np.min(pairwise_distances_argmin_min(reduced_data, centroids)[1], axis=1)
    
    # Determine Threshold for Anomalies
    # Set outliers_fraction
    # (e.g., 0.1 indicates 10% of points are considered outliers)
    outliers_fraction = 0.1
    number_of_outliers = int(len(distances) * outliers_fraction)
    
    # Find the threshold: minimum distance of the top "outliers_fraction" most distant points
    threshold = np.partition(distances, -number_of_outliers)[-number_of_outliers]
    
    # Mark Anomalies
    # Mark points as anomaly (1) or normal (0) based on the threshold
    anomalies = np.where(distances >= threshold, 1, 0)
    
    # Create a new column in the original DataFrame for anomalies
    df['Anomaly'] = anomalies
    
    # Visualize clusters with anomalies
    plt.figure(figsize=(10, 6))
    
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=labels, cmap='viridis', label="Normal Points")
    plt.scatter(df.iloc[anomalies == 1, 0], df.iloc[anomalies == 1, 1], color='red', label="Anomalies")
    plt.title('Clusters and Anomalies')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
    
