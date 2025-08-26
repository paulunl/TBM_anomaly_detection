# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 08:43:26 2020

@author: erharter/unterlass
"""


# script with utilities, formulas etc

from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
import pandas as pd
import PIL
from scipy import interpolate
from tqdm import tqdm

class utilities:

    def __init__(self):
        pass
    
    # function for the linear interpolation between existing observations to achieve an equal spacing of observations
    def equally_spaced_df(self, df, tunnellength, interval):
        min_length = df[tunnellength].min()
        max_length = df[tunnellength].max()

        equal_range = np.arange(min_length, max_length, interval)

        df_interp = pd.DataFrame({tunnellength: equal_range})

        for feature in df.drop(tunnellength, axis=1).columns:
            f = interpolate.interp1d(df[tunnellength], df[feature],
                                     kind='linear')
            df_interp[feature] = f(equal_range)

        df_interp.set_index(tunnellength, drop=False, inplace=True)

        return df_interp
    
    def check_for_miss_vals(self, df):
        print(df['Datum'])
        missing = df['Datum'].diff()
        missing = missing / np.timedelta64(1, 's')
        missing_idxs = np.where(missing > (60 * 60))[0]

        # print intervals where more than 1h of data are missing
        for missing_idx in missing_idxs:
            start = df['Datum'].iloc[missing_idx-1]
            stop = df['Datum'].iloc[missing_idx]
            print(f'missing data between {start} & {stop}')

        fig, ax = plt.subplots()
        ax.plot(df['Datum'], (missing / 60 / 60))
        ax.grid()
        ax.set_ylabel('missing data [h]')
        ax.tick_params(axis='x', labelrotation=45, )
        plt.tight_layout()

    def concat_tables(self, folder, drop_standstills=False,
                      check_for_miss_vals=True):

        filenames = []
        for file in listdir(folder):
            if file.split('.')[1] == 'csv':
                filenames.append(file)

        files = []

        for i, file in enumerate(filenames[:1000]):

            df = pd.read_csv(fr'{folder}\{file}', sep=';') #  encoding='latin1'
            print(file)
            print(f'{i} / {len(filenames)-1} csv done')  # status
            
            df = df.drop(df.columns[0], axis=1)
            column_names = ['Date', 'Stroke', 'Chainage [m]', 'Tunnel Distance [m]',
                            'Advance speed [mm/min]', 'Speed cutterhead [rpm]',
                            'Pressure advance cylinder bottom side [bar]',
                            'Torque cutterhead [MNm]',
                            'Total advance force [kN]', 'Penetration [mm/rot]',
                            'Pressure RSC left [bar]',
                            'Pressure RSC right[bar]',
                            'Path RSC left [mm]']

            df.columns = column_names

            df = df[(df['Tunnel Distance [m]'] >=0 | (df['Tunnel Distance [m]'].isnull()))] # getting rid of datapoints < 0 in Tunnel Distance or nan

            files.append(df)

        df = pd.concat(files, sort=True)

        df.dropna(inplace=True)

        # check for missing values in time series
        if check_for_miss_vals is True:
            self.check_for_miss_vals(df)

        return df, files

    def filter_outliers(self, df, mahal_features, percentile, length):
        drop_idxs = []

        for i in tqdm(df.index):
            #if i % 10000 == 0:
                #print(f'{i} / {len(df)}')
            if i > length and i <= len(df):
                window = df[i-length: i]
                # calcs the mahalanobis dist. for every point based on features
                comp = computation()
                try:
                    mahal = comp.MahalanobisDist(window[mahal_features].values)
                    thresh = np.percentile(mahal, percentile)
                    drop_idx = np.where(mahal > thresh)[0]
                    drop_idxs.append(window.index[drop_idx].values)
                except TypeError:
                    pass

        drop_idxs = np.concatenate(drop_idxs)
        df.drop(drop_idxs, inplace=True)
        df.index = np.arange(len(df))

        return df

###############################################################################

class computation:

    def __init__(self):
        pass

    def MahalanobisDist(self, data):
        data = np.transpose(data)
        #print(data.shape)
        n_dims = data.shape[0]
        # calculate the covariance matrix
        covariance_xyz = np.cov(data)
        # take the inverse of the covariance matrix
        try:
            inv_covariance_xyz = np.linalg.inv(covariance_xyz)

            means = []
            for i in range(n_dims):
                means.append(np.mean(data[i]))

            diffs = []
            for i in range(n_dims):
                diff = np.asarray([x_i - means[i] for x_i in data[i]])
                diffs.append(diff)
            diffs = np.transpose(np.asarray(diffs))

            # calculate the Mahalanobis Distance for each data sample
            md = []
            for i in range(len(diffs)):
                md.append(np.sqrt(np.dot(np.dot(np.transpose(diffs[i]),
                                                inv_covariance_xyz),
                                  diffs[i])))
            return md

        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                print('singular matrix')
                pass
            else:
                raise

    def s_en(self, thrust, area, rotations, torque, penetration):
        # computes the specific energy after Teale 1965
        f = thrust / area
        x = (2 * np.pi * rotations * torque) / (area * penetration)
        return (f + x)

    def s_pen(self, penetration, adv_force):
        # computes specific penetration
        spec_penetration = penetration / (adv_force / 1000)
        return spec_penetration

    def t_ratio(self, tot_cutters, r_cutter, M0,
                tot_adv_force, penetration,
                real_torque, cutter_positions):
        # computes the torque ratio after Radoncic 2014
        length = len(tot_adv_force)

        # avg. normal force per cutter
        Fn = (tot_adv_force / 1000) / tot_cutters
        Fn = np.meshgrid(np.arange(tot_cutters), Fn)[1]

        # cutting angle
        penetration = np.meshgrid(np.arange(tot_cutters),
                                  (penetration))[1]
        angle = np.degrees(np.arccos((r_cutter - penetration)/r_cutter))
        # avg. tangential force:
        Ft = Fn * np.tan(np.deg2rad(angle)/2)
        cutter_positions = np.meshgrid(cutter_positions, np.arange(length))[0]
        sums = (np.sum((cutter_positions * Ft), axis=1) + M0)           
        print(sums)
        theoretical_torque = sums

        torque_ratio = real_torque / theoretical_torque

        return torque_ratio, theoretical_torque

    def s_friction(self, thrust, cutterhead_force, tension_backupsys):
        return thrust - cutterhead_force - tension_backupsys
