 # -*- coding: TBM_Bf-8 -*-
"""
 TBM Operational Data-Driven Anomaly Detection in Hard Rock Excavations

---- link to paper
DOI: XXXX

Script containing the code of differnt anomaly detection techniques applied to
TBM operational data.

@author: Paul Unterlaß

"""
# =============================================================================
# Imports
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# =============================================================================
# Load Data Function
# =============================================================================
def load_data(tunnel, Class):    
    """
    Load and preprocess data based on the tunnel type (TBM_A or TBM_B).
    
    Parameters:
    - tunnel: The type of tunnel ('TBM_A' or 'TBM_B')
    - Class: The target class for classification
    
    Returns:
    - interval: Interval for test sets
    - test_start1, test_start2, test_start3: Start points for different test sets
    - test_end1, test_end2, test_end3: End points for different test sets
    - columns: Columns to be used in the model
    - df: Preprocessed DataFrame
    - df1, df2, df3: DataFrames split by test set intervals
    """
    if tunnel == 'TBM_B':
        # Load data and preprocess for TBM_B
        df = pd.read_parquet(fr'...')
        df['Class'] = df['Class'] - 1
        
        # Remove outliers based on machine limits
        df.drop(df[df['Torque cutterhead [MNm]'] > 10.2].index, inplace=True)
        df.drop(df[df['Total advance force [kN]'] > 27000].index, inplace=True)
        df.drop(df[df['Total advance force [kN]'] < 4000].index, inplace=True)
        df.drop(df[df['Penetration [mm/rot]'] < 0.1].index, inplace=True)
        df.reset_index(inplace=True, drop=True)
    
    elif tunnel == 'TBM_A':
        # Load data and preprocess for TBM_A
        df = pd.read_parquet(fr'..')
        
        # Remove outliers based on machine limits
        df.drop(df[df['Torque cutterhead [MNm]'] > 4.5].index, inplace=True)
        df.drop(df[df['Total advance force [kN]'] < 2000].index, inplace=True)
        df.drop(df[df['Total advance force [kN]'] > 17500].index, inplace=True)
        df.drop(df[df['Penetration [mm/rot]'] < 0.1].index, inplace=True)
        df = df[df['Tunnel Distance [m]'] > 1 * 1000]  # Exclude first km
        
        # Rename columns for consistency
        df.rename(columns={'GI': 'Class'}, inplace=True)
        df.rename(columns={'Speed cutterhead [rpm]': 'Speed cutterhead for display [rpm]'},
                  inplace=True)
        df.reset_index(inplace=True, drop=True)
    
    # Define the columns to be used
    columns = ['Tunnel Distance [m]',
               'Advance speed [mm/min]',
               'Penetration [mm/rot]',
               'Speed cutterhead for display [rpm]',
               'Torque cutterhead [MNm]',
               'Total advance force [kN]',
               'spec. penetration [mm/rot/MN]', 
               'torque ratio']
    
    rmc = ['Class']
    
    df = df[columns + rmc]  # Filter the required columns
    
    # Define train/validation/test sections for TBM_A and TBM_B
    if tunnel == 'TBM_A':
        # Column specifications and test section intervals for TBM_A
        columns = ['Advance speed [mm/min]',
                   'Penetration [mm/rot]',
                   'Speed cutterhead for display [rpm]',
                   'Torque cutterhead [MNm]',
                   'Total advance force [kN]',
                   'spec. penetration [mm/rot/MN]', 
                   'torque ratio']
        
        target_column = ['Class']
        
        # Test section intervals for TBM_A
        val_start, val_end = 10.5, 11.5
        test_start1, test_end1 = 3.5, 4.5
        test_start2, test_end2 = 5.0, 6.0
        test_start3, test_end3 = 9.5, 10.5
        interval = 0.05
        seq_size = 100  # Parameters for sequences
        
    elif tunnel == 'TBM_B':
        # Column specifications and test section intervals for TBM_B
        columns = ['Advance speed [mm/min]',
                   'Penetration [mm/rot]',
                   'Speed cutterhead for display [rpm]',
                   'Torque cutterhead [MNm]',
                   'Total advance force [kN]',
                   'spec. penetration [mm/rot/MN]', 
                   'torque ratio']
        
        df_spec_tor = df[['spec. penetration [mm/rot/MN]', 'torque ratio', 'Tunnel Distance [m]', 'Class']]
        target_column = ['Class']
        
        # Test section intervals for TBM_B
        val_start, val_end = 3.5, 4.5
        test_start1, test_end1 = 2.4, 2.8
        test_start2, test_end2 = 4.9, 5.25
        test_start3, test_end3 = 5.9, 6.2
        interval = 0.03
        seq_size = 100  # Parameters for sequences
    
    # Split data into different test sets based on tunnel distance
    df1 = df[(df['Tunnel Distance [m]'] >= test_start1 * 1000) & 
             (df['Tunnel Distance [m]'] < test_end1 * 1000)]
    
    df2 = df[(df['Tunnel Distance [m]'] >= test_start2 * 1000) & 
             (df['Tunnel Distance [m]'] < test_end2 * 1000)]
    
    df3 = df[(df['Tunnel Distance [m]'] >= test_start3 * 1000) & 
             (df['Tunnel Distance [m]'] < test_end3 * 1000)]
    
    return interval, test_start1, test_start2, test_start3, test_end1, test_end2, test_end3, columns, df, df1, df2, df3

# Load the data for TBM_A and a specific class (e.g., Class 3)
tunnel = 'TBM_A'
Class = 3

interval, test_start1, test_start2, test_start3, test_end1, test_end2, test_end3, columns, df, df1, df2, df3 = load_data(tunnel, Class)

# =============================================================================
# Isolation Forest for Anomaly Detection
# =============================================================================
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

def isolation_forest(df):
    """
    Detect anomalies in the data using Isolation Forest.
    
    Parameters:
    - df: DataFrame with the data to detect anomalies
    
    Returns:
    - df: DataFrame with added 'anomaly' column indicating anomalies
    """
    outliers_fraction = 0.02  # Fraction of outliers
    
    # Standardize the data
    scaler = StandardScaler()
    np_scaled = scaler.fit_transform(df[columns].values)
    data = pd.DataFrame(np_scaled, columns=columns)
    
    # Train Isolation Forest model
    model = IsolationForest(contamination=outliers_fraction)
    model.fit(data)
    
    # Add anomaly column to the DataFrame
    df['anomaly'] = model.predict(data)
    
    return df

# Apply Isolation Forest to the three test sets
df1 = isolation_forest(df1)
df2 = isolation_forest(df2)
df3 = isolation_forest(df3)

# =============================================================================
# Forecasting using VAR (Vector Auto Regression)
# =============================================================================

# Import necessary libraries
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

# Remove unnecessary columns from the dataframes for VAR analysis
df1_VAR = df1[columns]
df2_VAR = df2[columns]
df3_VAR = df3[columns]

# Check if the data is stationary using the Augmented Dickey-Fuller (ADF) Test
# The ADF test helps to verify if the time series data has constant statistical properties over time

# Function to perform ADF test
def adf_test(series):
    """
    Perform Augmented Dickey-Fuller test to check for stationarity.
    """
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

# Apply ADF test to each time series column in the datasets (df1, df2, df3)
for column in df1_VAR.columns:
    print(f"\nADF Test for {column}:")
    adf_test(df1_VAR[column])

for column in df2_VAR.columns:
    print(f"\nADF Test for {column}:")
    adf_test(df2_VAR[column])

for column in df3_VAR.columns:
    print(f"\nADF Test for {column}:")
    adf_test(df3_VAR[column])

# Function to detect anomalies using VAR (Vector Auto Regression)
def detect_anomalies_var(data):
    """
    Detect anomalies in multivariate time series using VAR model.
    
    Parameters:
    - data: pd.DataFrame, multivariate time series (columns are variables).
    - lag_order: int, lag order for the VAR model.
    - threshold: float, number of standard deviations for anomaly detection.
    
    Returns:
    - anomalies: pd.DataFrame, binary (1 = anomaly, 0 = normal) for each variable.
    - residuals: pd.DataFrame, residuals of the VAR model for each variable.
    """
    # Fit the VAR model to the data
    model = VAR(data)
    
    # Find the optimal lag order using AIC (Akaike Information Criterion)
    lag_order_results = model.select_order(maxlags=15)
    print(lag_order_results.summary())
    optimal_lag = lag_order_results.aic
    
    # Fit the model with the selected lag order
    results = model.fit(optimal_lag)
    
    # Get the predicted values from the model
    predicted = results.fittedvalues
    
    # Compute residuals (difference between actual and predicted values)
    residuals = data[optimal_lag:] - predicted
    
    # Initialize a DataFrame to store anomalies (binary values: 1 for anomaly, 0 for normal)
    anomalies = pd.DataFrame(0, index=residuals.index, columns=data.columns)
    
    # Set dynamic thresholds based on the 99th percentile of residuals for each variable
    for column in residuals.columns:
        threshold = residuals[column].quantile(0.99)
        anomalies[column] = (np.abs(residuals[column]) > threshold).astype(int)
    
    return anomalies, residuals

# Detect anomalies for each dataset (df1_VAR, df2_VAR, df3_VAR)
test1_anom, test1_resid = detect_anomalies_var(df1_VAR)
test2_anom, test2_resid = detect_anomalies_var(df2_VAR)
test3_anom, test3_resid = detect_anomalies_var(df3_VAR)

# Add additional columns ('Tunnel Distance [m]' and 'Class') to the VAR dataframes
df1_VAR['Tunnel Distance [m]'] = df1['Tunnel Distance [m]'] 
df1_VAR['Class'] = df1['Class']
df2_VAR['Tunnel Distance [m]'] = df2['Tunnel Distance [m]']
df2_VAR['Class'] = df2['Class']
df3_VAR['Tunnel Distance [m]'] = df3['Tunnel Distance [m]']
df3_VAR['Class'] = df3['Class']

# =============================================================================
# Clustering-based anomaly detection
# =============================================================================

from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min

# Load test datasets, drop 'Class' and 'Tunnel Distance [m]'
interval, test_start1, test_start2, test_start3, test_end1, test_end2, test_end3, columns, df, df1, df2, df3 = load_data(tunnel, Class)

# Remove unnecessary columns for clustering
df1_clust = df1.drop(columns=['Class', 'Tunnel Distance [m]'])
df2_clust = df2.drop(columns=['Class', 'Tunnel Distance [m]'])
df3_clust = df3.drop(columns=['Class', 'Tunnel Distance [m]'])

# Concatenate all datasets for overall clustering
df_all = pd.concat([df1, df2, df3], ignore_index=True)
df_all_clust = pd.concat([df1_clust, df2_clust, df3_clust], ignore_index=True)

# Dictionary containing the dataframes to perform clustering on
dfs = {
    "test_set1": df1_clust,
    "test_set2": df2_clust,
    "test_set3": df3_clust,
    'all_test_sets': df_all_clust
}

# Loop through the datasets and apply KMeans clustering
for test_set, df in dfs.items():
    # Scale the data using StandardScaler to standardize features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    # Define a range of cluster numbers for testing the optimal clusters
    n_clusters = range(1, 11)
    
    # Calculate inertia for different cluster numbers
    inertia = []
    for i in n_clusters:
        model = KMeans(n_clusters=i, random_state=42)
        model.fit(df_scaled)
        inertia.append(model.inertia_)
    
    # Use KneeLocator to find the optimal number of clusters (Elbow Method)
    knee_locator = KneeLocator(n_clusters, inertia, curve="convex", direction="decreasing")
    optimal_clusters = knee_locator.knee

    # Print the optimal number of clusters for each dataset
    print(f"The optimal number of clusters in {test_set} is: {optimal_clusters}")
    
    # Apply PCA for dimensionality reduction
    pca = PCA()
    pca.fit(df_scaled)
    
    # Calculate the explained variance ratio and the cumulative explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    
    # Automatically choose the number of components to keep for 90% explained variance
    n_components = np.argmax(cumulative_explained_variance >= 0.95) + 1
    print(f"Number of components to keep: {n_components}")
    
    # Apply PCA with the selected number of components
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(df_scaled)
    
    # Fit the KMeans model with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    kmeans.fit(reduced_data)
    
    # Get the cluster labels and centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    # Calculate distance from each point to its nearest centroid
    _, distances = pairwise_distances_argmin_min(reduced_data, centroids)
    
    # Set outlier fraction (e.g., 0.03 means 3% of points are considered outliers)
    outliers_fraction = 0.03
    number_of_outliers = int(len(distances) * outliers_fraction)
    
    # Determine the threshold for anomalies: the minimum distance of the top "outliers_fraction" most distant points
    threshold = np.partition(distances, -number_of_outliers)[-number_of_outliers]
    
    # Mark anomalies based on the threshold
    anomalies = np.where(distances >= threshold, 1, 0)
    
    # Create a new column in the original DataFrame to store anomaly results
    df['Anomaly'] = anomalies
    
    # (Optional) Visualize clusters with anomalies (commented out as plotting code)
    # plt.figure(figsize=(10, 6))
    # plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=labels, cmap='viridis', label="Normal Points")
    # plt.scatter(df.iloc[anomalies == 1, 0], df.iloc[anomalies == 1, 1], color='red', label="Anomalies")
    # plt.title('Clusters and Anomalies')
    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    # plt.legend()
    # plt.show()

# Re-add 'Tunnel Distance [m]' and 'Class' to each DataFrame
df1_clust['Tunnel Distance [m]'] = df1['Tunnel Distance [m]']
df1_clust['Class'] = df1['Class']

df2_clust['Tunnel Distance [m]'] = df2['Tunnel Distance [m]']
df2_clust['Class'] = df2['Class']

df3_clust['Tunnel Distance [m]'] = df3['Tunnel Distance [m]']
df3_clust['Class'] = df3['Class']

df_all_clust['Tunnel Distance [m]'] = df_all['Tunnel Distance [m]']
df_all_clust['Class'] = df_all['Class']
