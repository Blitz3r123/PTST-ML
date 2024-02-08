#==========================================================================================================================
#  ?                                                     ABOUT
#  @author         :  Kaleem Peeroo
#  @email          :  KaleemPeeroo@gmail.com
#  @repo           :  
#  @createdOn      :  2024-02-07
#  @description    :  Functions used for analysing results produced by linear regression and random forest models.
#==========================================================================================================================

import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import pytest
import sys

from pprint import pprint
from icecream import ic

logger = logging.getLogger(__name__)

#==========================================================================================================================
#                                                      CONSTANTS
#==========================================================================================================================
ALL_MODELS_DIRPATH = "./all_models/"

DDS_METRICS = [
    'avg_lost_samples',
    'avg_lost_samples_percentage',
    'avg_received_samples',
    'avg_received_samples_percentage',
    'avg_samples_per_sec',
    'avg_throughput_mbps',
    'latency_us',
    'total_lost_samples',
    'total_lost_samples_percentage',
    'total_received_samples',
    'total_received_samples_percentage',
    'total_samples_per_sec',
    'total_throughput_mbps'
]

STATS = [
    'min', 'max',
    'mean', 'std',
    '1', '2', '5', '10', 
    '25', '30', '40', '50', '60', '70', '75', '80', 
    '90', '95', '99'
]

STANDARDISATION_FUNCTIONS = ["none", "z_score", 'min_max', 'robust_scaler',]

TRANSFORM_FUNCTIONS = [
    "none",
    "log",
    "log10",
    "log2",
    "log1p",
    "sqrt",
]

ERROR_METRICS = [
    "rmse", 
    "mse", 
    "mae", 
    "mape", 
    "r2", 
    "medae", 
    "explained_variance"
]

#==========================================================================================================================
#                                                      FUNCTIONS
#==========================================================================================================================

def get_timestamps_from_filenames(filenames):
    timestamps = []

    for filename in filenames:
        if "results" not in filename:
            continue

        filename_items = filename.split("_")

        if len(filename_items) != 3:
            logger.error(f"Invalid file name: {filename}. When splitting by '_', expected 3 items. Got {len(filename_items)}: {filename_items}")
            return None
        
        date = filename_items[0]
        time = filename_items[1]

        timestamp = datetime.datetime.strptime(f"{date}_{time}", "%Y-%m-%d_%H-%M-%S")
        timestamps.append(timestamp)

    return timestamps        

def get_latest_result_file(result_csv_files):
    if len(result_csv_files) == 0:
        return None

    if len(result_csv_files) == 1:
        return result_csv_files[0]

    # Extract the date and time from the file names
    result_csv_timestamps = get_timestamps_from_filenames(result_csv_files)

    if result_csv_timestamps is None:
        return None

    if len(result_csv_timestamps) == 0:
        logger.error("No valid timestamps found in file names.")
        return None

    # Sort the files by date and time
    result_csv_timestamps.sort(reverse=True)

    # Return the latest file
    latest_timestamp = result_csv_timestamps[0]
    latest_timestamp_str = latest_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    latest_file = [f for f in result_csv_files if latest_timestamp_str in f][0]

    return latest_file

def get_model_results():
    curr_files = os.listdir("./")
    if len(curr_files) == 0:
        logger.error("No files found in current directory.")
        return None
    
    csv_files = [f for f in curr_files if f.endswith(".csv")]
    if len(csv_files) == 0:
        logger.error("No CSV files found in current directory.")
        return None
    
    result_csv_files = [f for f in csv_files if "results" in f]
    if len(result_csv_files) == 0:
        logger.error("No results CSV files found in current directory.")
        return None
    
    if len(result_csv_files) > 1:
        result_csv_file = get_latest_result_file(result_csv_files)
    else:
        result_csv_file = result_csv_files[0]

    if result_csv_file is None:
        logger.error("No valid results CSV file found.")
        return pd.DataFrame()

    return pd.read_csv(result_csv_file)

def is_df_valid(df):
    # What am I looking for?
    # [ ] Does df['metric_of_interest'].unique() and DDS_METRICS match
    # [ ] Are there any NaNs in the dataframe?
    # [ ] Are there any duplicates in the dataframe?
    # [ ] Group by model_type and int_or_ext and count the number of rows to make sure they all match
    # [ ] Does df['standardisation'].unique() and STANDARDISATION_FUNCTIONS match
    # [ ] Does df['transform_function'].unique() and TRANSFORM_FUNCTIONS match
    # [ ] Does df['error_type'].unique() and ['rmse', 'mse', 'mae', 'mape', 'r', 'medae', 'explained_variance'] match
    # [ ] Does df['model_type'].unique() and ['linear_regression', 'random_forest'] match
    # [ ] Does df['int_or_ext'].unique() and ['interpolation', 'extrapolation'] match

    if df is None:
        logger.error("Empty dataframe.")
        return False

    if len(df) == 0:
        logger.error("Empty dataframe.")
        return False

    if len(df['metric_of_interest'].unique()) != len(DDS_METRICS):
        logger.error(f"Unique metrics in dataframe: {df['metric_of_interest'].unique()}. Expected: {DDS_METRICS}")
        return False
    
    if df.isna().sum().sum() > 0:
        logger.error(f"NaNs found in dataframe.")
    
    if df.duplicated().sum() > 0:
        logger.error(f"Duplicates found in dataframe.")
        return False
    
    if df.groupby(['model_type', 'int_or_ext']).size().nunique() != 1:
        logger.error(f"Number of rows for each group of model_type and int_or_ext do not match.")
        return False
    
    if len(df['standardisation_function'].unique()) != len(STANDARDISATION_FUNCTIONS):
        logger.error(f"Unique standardisation functions in dataframe: {df['standardisation_function'].unique()}. Expected: {STANDARDISATION_FUNCTIONS}")
        return False
    
    if len(df['transform_function'].unique()) != len(TRANSFORM_FUNCTIONS):
        logger.error(f"Unique transform functions in dataframe: {df['transform_function'].unique()}. Expected: {TRANSFORM_FUNCTIONS}")
        return False
    
    if len(df['error_type'].unique()) != len(ERROR_METRICS):
        logger.error(f"Unique error types in dataframe: {df['error_type'].unique()}. Expected: {ERROR_METRICS}")
        return False
    
    if len(df['model_type'].unique()) != 2:
        logger.error(f"Unique model types in dataframe: {df['model_type'].unique()}. Expected: ['Linear Regresssion', 'Random Forests']")
        return False
    
    if len(df['int_or_ext'].unique()) != 2:
        logger.error(f"Unique int_or_ext in dataframe: {df['int_or_ext'].unique()}. Expected: ['interpolation', 'extrapolation']")
        return False

    return True

def main():
    logging.basicConfig(level=logging.WARNING)
    df = get_model_results()

    if not is_df_valid(df):
        return

if __name__ == "__main__":
    if pytest.main(["-q", "./"]) == 0:
        main()
    else:
        logger.error("Tests failed.")
        sys.exit(1)