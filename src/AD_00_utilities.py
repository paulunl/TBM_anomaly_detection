# -*- coding: utf-8 -*-
"""
TBM Operational Data-Driven Anomaly Detection in Hard Rock Excavations

Script containing utilities, formulas, etc.

@author: Georg Erharter / Paul Unterlaß
"""

from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
import pandas as pd
import PIL
from scipy import interpolate
from tqdm import tqdm

class Utilities:
    """
    A utility class for processing tunneling data, including interpolation,
    missing value detection, and data concatenation.
    """
    
    def __init__(self):
        pass
    
    def equally_spaced_df(self, df, tunnellength, interval):
        """
        Perform linear interpolation to achieve equal spacing of observations.
        
        :param df: Input DataFrame.
        :param tunnellength: Column name representing tunnel length.
        :param interval: Desired spacing interval.
        :return: DataFrame with equally spaced observations.
        """
        min_length = df[tunnellength].min()
        max_length = df[tunnellength].max()
        equal_range = np.arange(min_length, max_length, interval)
        
        df_interp = pd.DataFrame({tunnellength: equal_range})
        
        for feature in df.drop(columns=[tunnellength]):
            f = interpolate.interp1d(df[tunnellength], df[feature], kind='linear')
            df_interp[feature] = f(equal_range)
        
        df_interp.set_index(tunnellength, drop=False, inplace=True)
        return df_interp
    
    def check_for_miss_vals(self, df):
        """
        Check for missing values in the 'Datum' column and plot missing intervals.
        
        :param df: Input DataFrame with a 'Datum' column.
        """
        missing = df['Datum'].diff() / np.timedelta64(1, 's')
        missing_idxs = np.where(missing > 3600)[0]  # Identify gaps greater than 1 hour

        for idx in missing_idxs:
            print(f'Missing data between {df["Datum"].iloc[idx-1]} & {df["Datum"].iloc[idx]}')
        
        plt.figure()
        plt.plot(df['Datum'], missing / 3600)
        plt.grid()
        plt.ylabel('Missing Data (hours)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
    def filter_outliers(self, df, mahal_features, percentile, length):
        """
        Remove outliers using the Mahalanobis distance.
        
        :param df: Input DataFrame.
        :param mahal_features: Features used for Mahalanobis distance calculation.
        :param percentile: Threshold percentile for outlier removal.
        :param length: Window size for computing Mahalanobis distance.
        :return: DataFrame with outliers removed.
        """
        drop_idxs = []
        
        for i in tqdm(df.index):
            if i > length:
                window = df.iloc[i-length:i]
                comp = Computation()
                try:
                    mahal = comp.MahalanobisDist(window[mahal_features].values)
                    thresh = np.percentile(mahal, percentile)
                    drop_idxs.append(window.index[mahal > thresh].values)
                except TypeError:
                    pass
        
        df.drop(np.concatenate(drop_idxs), inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

class Computation:
    """
    Class for mathematical computations related to tunneling data.
    """
    
    def MahalanobisDist(self, data):
        """
        Compute Mahalanobis distance for anomaly detection.
        """
        data = np.transpose(data)
        try:
            inv_cov = np.linalg.inv(np.cov(data))
            means = np.mean(data, axis=1)
            diffs = data - means[:, None]
            return np.sqrt(np.sum(np.dot(inv_cov, diffs) * diffs, axis=0))
        except np.linalg.LinAlgError:
            print('Singular matrix encountered')
            return []
    
    def s_en(self, thrust, area, rotations, torque, penetration):
        """
        Compute specific energy (Teale 1965).
        """
        return (thrust / area) + ((2 * np.pi * rotations * torque) / (area * penetration))
    
    def s_pen(self, penetration, adv_force):
        """
        Compute specific penetration.
        """
        return penetration / (adv_force / 1000)
    
    def t_ratio(self, tot_cutters, r_cutter, M0, tot_adv_force, penetration, real_torque, cutter_positions):
        """
        Compute torque ratio (Radoncic 2014).
        """
        Fn = (tot_adv_force / 1000) / tot_cutters  # Normal force per cutter
        angle = np.degrees(np.arccos((r_cutter - penetration) / r_cutter))
        Ft = Fn * np.tan(np.deg2rad(angle) / 2)  # Tangential force
        sums = np.sum(cutter_positions * Ft, axis=1) + M0
        return real_torque / sums, sums
