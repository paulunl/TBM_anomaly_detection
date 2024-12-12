# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 08:40:09 2020

@author: unterlass
"""
'''
Concat: Function to read the single .csv files containing the raw TBM data and merging
them into a single datasets. With the options to drop data record during
standstills of the machine and a check for missing values.

Preprocessor: Function containing a pre-processing routine for TBM data

'''
# =============================================================================
# Read single .csv files and concatenate
# =============================================================================

def concat():
    from os import listdir
    import pandas as pd

    # import raw data and concat individual excel files to big parquet file/csv
    folder = (r'M:\FMT-A\4000_Forschung\4400_Proj_ab_Okt2018'
    r'\002_Laufende_Proj\ngProj\TBM_Daten\BBT\bbt-ftp\BBT Daten'
    r'\EKS TVM\Maschienendaten\formatted')

    drop_standstills = True  # set to true if standstills should be dropped
    check_for_miss_vals = False  # set to true, it then generates plot of missing values in dataset
    
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
    
    df, files = concat_tables(folder, drop_standstills=drop_standstills,  # loads the concat_tables function from the utils
                              check_for_miss_vals=check_for_miss_vals)   # script and excutes it

    if drop_standstills is False:
        df.to_parquet(r'03_data\00_TBMdata_wStandstills.gzip', index=False)  # saves the concatenated raw data with standstills
        # df.to_csv(fr'01_processed_data\00_TBMdata_wStandstills.csv', index=False) # as parquet or csv file in the working directory
    else:
        df.to_parquet(r'03_data\01_TBMdata.gzip', index=False)  # same as aboth without standstill data
        # df.to_csv(fr'01_processed_data\00_TBMdata.csv', index=False)

    return df

df = concat()

# =============================================================================
# TBM operational data pre-processing routine
# =============================================================================

def preprocessor():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from VAE_00_utilities import utilities, computation # import utilities and computation classes from the utilities script
    
    # define variables for the imported classes
    utils = utilities()
    comp = computation()
    
    ###########################################################################
    # load merged csv files containing the raw data
    df = pd.read_parquet(r'03_data\01_TBMdata.gzip') # pd.read_parquet reads parquet files
    
    # drop all other columns
    df = df.loc[:,['Date', 'Stroke', 'Chainage [m]', 'Tunnel Distance [m]',
                    'Advance speed [mm/min]', 'Speed cutterhead [rpm]',
                    'Pressure advance cylinder bottom side [bar]',
                    'Torque cutterhead [MNm]', 'Total advance force [kN]',
                    'Penetration [mm/rot]']
                ] # input column name(s) which you do not want to drop
    
    print('# datapoints without standstills', df['Tunnel Distance [m]'].count()) # prints the number of datapoints
    
    ###########################################################################
    # calculate the median for observations with identical positions
    df_median = df.groupby('Tunnel Distance [m]', as_index = False).median() # groups the dataset by tunneldistance and calculates the median for observations with identical positions in place
    print('\n# datapoints after grouping by Tunnel Distance', df_median['Tunnel Distance [m]'].count()) # prints the number of datapoints after grouping
    
    '''
    # plot histogram for points with and without identical position
    df_grouped = df.groupby(['Tunnel Distance [m]'], as_index=True).agg(['count'])
    df_median_grouped = df_median.groupby(['Tunnel Distance [m]'], as_index=True).agg(['count']) #df_median.groupby(['Tunnel Distance [m]']).agg(['count']).index.values
    
    fig, (ax) = plt.subplots()
    ax.hist(df_grouped.index.values, 50, fc='darkgray', histtype='barstacked', ec='black')
    ax.hist(df_median_grouped.index.values, 50, fc='orange', histtype='barstacked', ec='black')
    ax.set_xlim(0, 6895)
    ax.set_xlabel('Tunnel Distance [m]')
    ax.set_ylabel('n', rotation=0)
    plt.tight_layout()
    plt.savefig(r'02_plots\hist_groupedbyTM.png', dpi=600)
    '''
    ###########################################################################
    # mahalanobis distance based outlier filtering
    # features to be filtered
    mahal_features = ['Advance speed [mm/min]',
                      'Speed cutterhead [rpm]',
                      'Pressure advance cylinder bottom side [bar]',
                      'Torque cutterhead [MNm]',
                      'Total advance force [kN]',
                      'Penetration [mm/rot]',
                    ]
    
    # filter outliers with mahal distribution, for every datapoint with respect to the previous 100,
    # deleting the ones that lye out of the P85 percentile
    df_mahal = utils.filter_outliers(df_median, mahal_features, 85, 100) # calls filter_outliers function from the utilities
    
    df_mahal = df_mahal.sort_values(by=['Tunnel Distance [m]']) # sorting the resulting observations by tunnel distance
    df_mahal.index = np.arange(len(df_mahal)) # updates the index of the dataframe
    print('# datapoints after mahal distr.', df_mahal['Tunnel Distance [m]'].count()) # prints the number of datapoints after filtering
    
    ###########################################################################
    # linear interpolation between existing observations to achieve an equal spacing of observations
    # difference in Tunnel Distance between single data points
    interval = df_mahal['Tunnel Distance [m]'].diff().median() # calculates the median interval between observations
    interval = round(interval, 2) # rounds the interval
    df_equal = utils.equally_spaced_df(df_mahal, 'Tunnel Distance [m]', interval) # calls the equally_spaced function from the utilities
    print('# datapoints after linear interp.', df_equal['Tunnel Distance [m]'].count()) # prints the number of datapoints after the linear interpolation
    
    df = df_equal
    
    ###########################################################################
    # calculate spec. penetration
    penetration = df['Penetration [mm/rot]']
    tot_adv_force = df['Total advance force [kN]']
    
    df['spec. penetration [mm/rot/MN]'] = comp.s_pen(penetration, tot_adv_force)
    
    # calculate torque ratio
    penetration = df['Penetration [mm/rot]']
    tot_cutters = 46
    r_cutter = 241.3 # [mm] 19" == 48.26cm
    # calculate M0 (torque needed for turning the cutting wheel when no advance)
    # df_M0 = df_ws.loc[(df_ws['Speed [mm/min]'] <=0) &
    #                   (df_ws['CH Rotation [rpm]'] !=0) &
    #                   (df_ws['Thrust Force [kN]'] <=0)] #&
                      # (df_ws['Pressure A']>=40) &
                      # (df_ws['Pressure A']<=80)]
    
    M0 = df['Torque cutterhead [MNm]'].mean()*1000
    real_torque = df['Torque cutterhead [MNm]']*1000
    cutter_positions = 0.3*(7.93/2)
    
    df['torque ratio'], df['theoretical torque [MNm]'] = comp.t_ratio(
        tot_cutters, r_cutter, M0, tot_adv_force, penetration,
        real_torque, cutter_positions)
    
    ###########################################################################
    # add class labels
    file = (r'M:\FMT-A\4000_Forschung\4400_Proj_ab_Okt2018\002_Laufende_Proj'
    r'\ngProj\TBM_Daten\BBT\bbt-Sicherung_Erharter\BBT_labels'
    r'\Indikation_Geologie_aus_geol_Doku_Übersicht.csv')
            
    df_labels = pd.read_csv(file, sep=';', header=[0], dtype=np.float64,
                            decimal=',')
    
    df_labels = df_labels.iloc[1::2, :]  # delete every second row because of the format in the excel file
    df_labels.set_index('Tunnelmeter', inplace=True)
    
    df = pd.merge(df, df_labels,
                  left_index=True, right_index=True, how='outer') # merge labels with TBM data
    
    df['GI'].fillna(method='bfill', inplace=True) # filling the gaps with labels
    df['GI'] = df['GI'] - 1  # subtract 1, for later one-hot encoding
    
    ###########################################################################
    #save df
    df = df.round({'Tunnel Distance [m]':2})
    df = df.dropna() # drop rows with NaN values
    df.reset_index(inplace=False)  # resets the index
    df.to_parquet(r'03_data\02_TBMdata_preprocessed.gzip', index=False) # save the preprocessed data

    ###########################################################################
    return df

df = preprocessor()