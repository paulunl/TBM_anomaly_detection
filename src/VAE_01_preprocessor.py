# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 08:40:09 2020

@author: unterlass
"""
'''
Function to read the single .csv files containing the raw TBM data and merging
them into a single datasets. With the options to drop data record during
standstills of the machine and a check for missing values.
'''


def concat():

    from VAE_00_utilities import utilities # import utilities class and its functions defined in the utilities script
    utils = utilities()  # define variable for the utilities class

    # import raw data and concat individual excel files to big parquet file/csv
    folder = (r'M:\FMT-A\4000_Forschung\4400_Proj_ab_Okt2018'
    r'\002_Laufende_Proj\ngProj\TBM_Daten\BBT\bbt-ftp\BBT Daten'
    r'\EKS TVM\Maschienendaten\formatted')

    drop_standstills = True  # set to true if standstills should be dropped
    check_for_miss_vals = False  # set to true, it then generates plot of missing values in dataset

    df, files = utils.concat_tables(folder, drop_standstills=drop_standstills,  # loads the concat_tables function from the utils
                                    check_for_miss_vals=check_for_miss_vals)   # script and excutes it

    if drop_standstills is False:
        df.to_parquet(r'03_data\00_TBMdata_wStandstills.gzip', index=False)  # saves the concatenated raw data with standstills
        # df.to_csv(fr'01_processed_data\00_TBMdata_wStandstills.csv', index=False) # as parquet or csv file in the working directory
    else:
        df.to_parquet(r'03_data\01_TBMdata.gzip', index=False)  # same as aboth without standstill data
        # df.to_csv(fr'01_processed_data\00_TBMdata.csv', index=False)

    return df
