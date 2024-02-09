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
    'mean', 'std',
    'min', 'max',
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

def format_stats(stats):
    if len(stats) == 0:
        return []
    
    formatted_stats = []

    for stat in stats:
        if stat is None:
            continue
        if stat == "":
            continue
        if len(stat) == 0:
            continue
        if len(stat.strip()) == 0:
            continue
        
        try:
            stat = int(stat)
            if stat % 10 == 1:
                formatted_stats.append(f"{stat}st")
            elif stat % 10 == 2:
                formatted_stats.append(f"{stat}nd")
            elif stat % 10 == 3:
                formatted_stats.append(f"{stat}rd")
            else:
                formatted_stats.append(f"{stat}th")
        except:
            
            if stat.lower() == "std":
                formatted_stats.append("std")
            else:
                formatted_stats.append(stat.capitalize())

    return formatted_stats

def get_table_columns(error_metrics):
    columns = []

    for error_metric in error_metrics:
        if error_metric is None:
            continue
        if error_metric == "":
            continue
        if len(error_metric) == 0:
            continue
        if len(error_metric.strip()) == 0:
            continue
        if "train" in error_metric.lower():
            continue
        if "test" in error_metric.lower():
            continue

        try:
            error_metric = int(error_metric)
            continue
        except:
            pass

        columns.append(f"{error_metric.upper()} Train")
        columns.append(f"{error_metric.upper()} Test")

    return columns

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
    if df is None:
        logger.error("Empty dataframe.")
        return False

    if len(df) == 0:
        logger.error("Empty dataframe.")
        return False

    # [ ] Does df['metric_of_interest'].unique() and DDS_METRICS match
    if len(df['metric_of_interest'].unique()) != len(DDS_METRICS):
        logger.error(f"Unique metrics in dataframe: {df['metric_of_interest'].unique()}. Expected: {DDS_METRICS}")
        return False
    
    # [ ] Are there any NaNs in the dataframe?
    if df.isna().sum().sum() > 0:
        logger.error(f"NaNs found in dataframe.")
    
    # [ ] Are there any duplicates in the dataframe?
    if df.duplicated().sum() > 0:
        logger.error(f"Duplicates found in dataframe.")
        return False
    
    # [ ] Group by model_type and int_or_ext and count the number of rows to make sure they all match
    if df.groupby(['model_type', 'int_or_ext']).size().nunique() != 1:
        logger.error(f"Number of rows for each group of model_type and int_or_ext do not match.")
        return False
    
    # [ ] Does df['standardisation'].unique() and STANDARDISATION_FUNCTIONS match
    if len(df['standardisation_function'].unique()) != len(STANDARDISATION_FUNCTIONS):
        logger.error(f"Unique standardisation functions in dataframe: {df['standardisation_function'].unique()}. Expected: {STANDARDISATION_FUNCTIONS}")
        return False
    
    # [ ] Does df['transform_function'].unique() and TRANSFORM_FUNCTIONS match
    if len(df['transform_function'].unique()) != len(TRANSFORM_FUNCTIONS):
        logger.error(f"Unique transform functions in dataframe: {df['transform_function'].unique()}. Expected: {TRANSFORM_FUNCTIONS}")
        return False

    # [ ] Does df['error_type'].unique() and ['rmse', 'mse', 'mae', 'mape', 'r2', 'medae', 'explained_variance'] match
    if len(df['error_type'].unique()) != len(ERROR_METRICS):
        logger.error(f"Unique error types in dataframe: {df['error_type'].unique()}. Expected: {ERROR_METRICS}")
        return False
    
    # [ ] Does df['model_type'].unique() and ['linear_regression', 'random_forest'] match
    if len(df['model_type'].unique()) != 2:
        logger.error(f"Unique model types in dataframe: {df['model_type'].unique()}. Expected: ['Linear Regresssion', 'Random Forests']")
        return False
    
    # [ ] Does df['int_or_ext'].unique() and ['interpolation', 'extrapolation'] match
    if len(df['int_or_ext'].unique()) != 2:
        logger.error(f"Unique int_or_ext in dataframe: {df['int_or_ext'].unique()}. Expected: ['interpolation', 'extrapolation']")
        return False

    return True

def main():
    logging.basicConfig(level=logging.WARNING)
    df = get_model_results()

    if not is_df_valid(df):
        return
    
    df = df.sort_values(
        by=[
            'model_type', 
            'int_or_ext', 
            'metric_of_interest', 
            'standardisation_function', 
            'transform_function', 
            'error_type'
        ],
        ascending=[
            True, 
            False, 
            True, 
            True, 
            True, 
            True
        ]
    )

    # df is ready to analyse and turn into latex tables here
    
    '''
    What to do?
    1. Group by model_type and int_or_ext, and metric_of_interest.
    2. Group further by standardisation_function and transform_function.
    3. For each std+tfm group create a table.
        3.1. Table should have first column as distribution stats (min, max, mean, std, 1, 2, 5, 10, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99), and the rest of the columns as pairs of train and test errors for each error type.
    '''

    df_grouped_by_model_type_int_or_ext_metric_of_interest = df.groupby(['model_type', 'int_or_ext', 'metric_of_interest'])

    for (model_type, int_or_ext, metric_of_interest), first_group in df_grouped_by_model_type_int_or_ext_metric_of_interest:
        
        df_grouped_by_std_tfm = first_group.groupby(['standardisation_function', 'transform_function'])

        for (std, tfm), second_group in df_grouped_by_std_tfm:
            
            table_columns = get_table_columns(ERROR_METRICS)
            table = pd.DataFrame(columns=["Distribution Statistic"] + table_columns)

            stats = format_stats(STATS)

            for stat in STATS:
                
                stat_index = STATS.index(stat)
                formatted_stat = stats[stat_index]
                stat_row = [formatted_stat]

                for error_type in ERROR_METRICS:
                    output_variable = f"{metric_of_interest}_{stat}"
                    
                    train_error = second_group[
                        (second_group['error_type'] == error_type) & 
                        (second_group['output_variable'] == output_variable)
                    ]['train_error']
                    
                    test_error = second_group[
                        (second_group['error_type'] == error_type) & 
                        (second_group['output_variable'] == output_variable)
                    ]['test_error']
                    
                    if len(train_error) == 0:
                        train_error = np.nan

                    if len(test_error) == 0:
                        test_error = np.nan

                    train_value = train_error.values[0]
                    test_value = test_error.values[0]

                    train_value = "{:,.3f}".format(train_value)
                    test_value = "{:,.3f}".format(test_value)

                    stat_row.append(train_value)
                    stat_row.append(test_value)

                table = pd.concat([table, pd.DataFrame([stat_row], columns=["Distribution Statistic"] + table_columns)], ignore_index=True)

            print(table.to_latex(index=False))
            asdf

if __name__ == "__main__":
    if pytest.main(["-q", "./"]) == 0:
        main()
    else:
        logger.error("Tests failed.")
        sys.exit(1)