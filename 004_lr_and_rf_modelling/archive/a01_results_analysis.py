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
import logging

from pprint import pprint
from icecream import ic
from rich.progress import track

logger = logging.getLogger(__name__)

from constants import *

#==========================================================================================================================
#                                                      FUNCTIONS
#==========================================================================================================================

def get_train_test_errors(df, error_type, output_variable):
    if df is None:
        logger.error("Empty dataframe passed to get_train_test_errors")
        return None, None
    if error_type is None:
        logger.error("None error_type passed to get_train_test_errors")
        return None, None
    if output_variable is None:
        logger.error("None output_variable passed to get_train_test_errors")
        return None, None
    if len(df) == 0:
        logger.error("Empty dataframe passed to get_train_test_errors")
        return None, None

    required_columns = ['error_type', 'output_variable', 'train_error', 'test_error']
    if not all(col in df.columns for col in required_columns):
        logger.error(f"Required columns not found in dataframe: {required_columns}. Found: {df.columns}")
        return None, None

    train_error = df[
        (df['error_type'] == error_type) & 
        (df['output_variable'] == output_variable)
    ]['train_error']
    
    test_error = df[
        (df['error_type'] == error_type) & 
        (df['output_variable'] == output_variable)
    ]['test_error']
    
    train_error = train_error.values[0] if len(train_error) > 0 else None
    test_error = test_error.values[0] if len(test_error) > 0 else None
        
    return train_error, test_error

def format_stat(stat):
    if stat is None:
        return None
    if stat == "":
        return None
    if len(stat) == 0:
        return None
    if len(stat.strip()) == 0:
        return None
    
    try:
        stat = int(stat)
        if stat % 10 == 1:
            return f"{stat}st"
        elif stat % 10 == 2:
            return f"{stat}nd"
        elif stat % 10 == 3:
            return f"{stat}rd"
        else:
            return f"{stat}th"
    except:
        if stat.lower() == "std":
            return "std"
        else:
            return stat.capitalize()

def format_stats(stats):
    if len(stats) == 0:
        return []
    
    formatted_stats = []
    for stat in stats:
        if stat:
            formatted_stats.append(format_stat(stat))

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

        if "explained_variance" in error_metric.lower():
            error_metric = "Explained Variance"
            columns.append(f"{error_metric} Train")
            columns.append(f"{error_metric} Test")
        else:
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
        logger.warning(f"NaNs found in dataframe.")
    
    # [ ] Are there any duplicates in the dataframe?
    if df.duplicated().sum() > 0:
        logger.warning(f"Duplicates found in dataframe.")
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

def get_error_metric_df_for_stats(error_metrics, metric_of_interest, group):

    if error_metrics is None:
        return None, None
    
    if metric_of_interest is None:
        return None, None
    
    if group is None:
        return None, None
    
    if len(group) == 0:
        return None, None
    
    error_metrics = [e for e in error_metrics if e is not None and e != "" and len(e) > 0 and len(e.strip()) > 0]
    if len(error_metrics) == 0:
        return None, None
    

    table_columns = get_table_columns(error_metrics)
    table = pd.DataFrame(columns=["Distribution Statistic"] + table_columns)
    column_formats = "|c|" + "r|" * len(table_columns)

    for stat in STATS:
        
        stat_row = [ format_stat(stat) ]
        
        for error_type in error_metrics:
            output_variable = f"{metric_of_interest}_{stat}"
            
            train_error, test_error = get_train_test_errors(
                group, 
                error_type, 
                output_variable
            )

            if train_error is None or test_error is None:
                stat_row.append("N/A")
                stat_row.append("N/A")
            else:
                stat_row.append("{:,.3f}".format(train_error))
                stat_row.append("{:,.3f}".format(test_error))

        table = pd.concat(
            [
                table,
                pd.DataFrame(
                    [stat_row], 
                    columns=["Distribution Statistic"] + table_columns
                )
            ], 
            ignore_index=True
        )

    return table, column_formats

def generate_latex_table_for_error_metrics(error_metrics, metric_of_interest, group, model_type, int_or_ext, std, tfm):
    metric_of_interest_string = metric_of_interest.replace("_", "\\_")
    std_string = std.replace("_", "\\_")
    tfm_string = tfm.replace("_", "\\_")

    table, column_formats = get_error_metric_df_for_stats(error_metrics, metric_of_interest, group)

    table_label = f"tab:{model_type.replace(' ', '_')}_{int_or_ext.capitalize()}_{metric_of_interest}_{std}_{tfm}"
    table_caption = f"{model_type} {int_or_ext.capitalize()} {metric_of_interest_string} Standardisation: {std_string} Transformation: {tfm_string}"
    latex_table = table.to_latex(index=False, caption=table_caption, label=table_label, column_format=column_formats) + "\n"
    latex_table = latex_table.replace("\\begin{table}", "\\begin{landscape}\n\\begin{table}")
    latex_table = latex_table.replace("\\end{table}", "\\end{landscape}\n\\end{table}")

    return latex_table

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

    '''
    What to do?
    1. Group by model_type and int_or_ext, and metric_of_interest.
    2. Group further by standardisation_function and transform_function.
    3. For each std+tfm group create a table.
        3.1. Table should have first column as distribution stats (min, max, mean, std, 1, 2, 5, 10, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99), and the rest of the columns as pairs of train and test errors for each error type.
    '''

    df_grouped_by_model_type_int_or_ext_metric_of_interest = df.groupby(['model_type', 'int_or_ext', 'metric_of_interest'], sort=False)

    latex_output = ""

    for (model_type, int_or_ext, metric_of_interest), first_group in track(df_grouped_by_model_type_int_or_ext_metric_of_interest, description="Processing..."):

        metric_of_interest_string = metric_of_interest.replace("_", "\\_")

        latex_output += f"\\subsection{{{model_type} {int_or_ext.capitalize()} {metric_of_interest_string}}}\n"

        df_grouped_by_std_tfm = first_group.groupby(['standardisation_function', 'transform_function'])

        for (std, tfm), second_group in df_grouped_by_std_tfm:

            std_string = std.replace("_", "\\_")
            tfm_string = tfm.replace("_", "\\_")
            latex_output += f"\\subsubsection{{{std_string} {tfm_string}}}\n"

            # Pair the following error metrics together:
            # r2, rmse
            # mae, medae
            # mse, mape
            # explained_variance

            error_metric_pairs = [
                ['r2', 'rmse'],
                ['mae', 'medae'],
                ['mse', 'mape'],
                ['explained_variance']
            ]

            for error_metrics in error_metric_pairs:
                latex_output += generate_latex_table_for_error_metrics(error_metrics, metric_of_interest, second_group, model_type, int_or_ext, std, tfm)

    with open("output.tex", "w") as f:
        f.write(latex_output)

if __name__ == "__main__":
    if pytest.main(["-q", "./", "--exitfirst"]) == 0:
        main()
    else:
        logger.error("Tests failed.")
        sys.exit(1)