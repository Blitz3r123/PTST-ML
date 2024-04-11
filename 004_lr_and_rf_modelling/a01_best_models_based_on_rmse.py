import pandas as pd
import pytest
import os
import sys

import logging
logger = logging.getLogger(__name__)

from pprint import pprint

def get_csv_files(dirname="./"):
   if not os.path.exists(dirname):
        logger.error(f"{dirname} doesnt exist")
        return []

   if dirname is None:
        logger.error("No dirname passed to get_csv_files")
        return [] 

   return [f for f in os.listdir(dirname) if f.endswith('.csv') and 'results' in f and os.path.isfile(f)]

def get_results_df(dirname="./"):
    if not os.path.exists(dirname):
        logger.error(f"{dirname} doesnt exist")
        return None

    files_in_dir = os.listdir(dirname)
    if len(files_in_dir) == 0:
        logger.error(f"No files found in {dirname}")
        return None

    csv_files = get_csv_files(dirname)
    if csv_files is None:
        logger.error(f"No csv files found in {dirname}")
        return None

    if len(csv_files) == 0:
        logger.error(f"No csv files found in {dirname}")
        return None

    results_csv = csv_files[0]

    return pd.read_csv(results_csv)

def calculate_average_metrics_from_models(group_df=pd.DataFrame()):
    if group_df is None:
        logger.error(f"No dataframe passed to calculate_average_metrics_from_models().")
        return None

    if len(group_df.columns) == 0:
        logger.error(f"No columns found in dataframe for calculate_average_metrics_from_models()")
        return None

    wanted_cols = ['standardisation_function', 'transform_function', 'r2_test_error', 'rmse_test_error']
    df_cols = group_df.columns

    mismatching_cols = list(
        set(df_cols) - set(wanted_cols)
    )

    if len(mismatching_cols) >= 0:
        logger.error(f"Mismatch between df cols and wanted cols for calculate_average_metrics_from_models().\n\tWanted: {wanted_cols}\n\tFound: {df_cols}")
        return None

    avg_metric_per_models = []

    for std_df, std_tf_group_df in group_df.groupby([
        'standardisation_function',
        'transform_function'
    ]):
        r2_test_col = std_tf_group_df['r2_test_error']
        rmse_test_col = std_tf_group_df['rmse_test_error']

        avg_r2_test = r2_test_col.mean()
        avg_rmse_test = rmse_test_col.mean()

        standardisation_function = std_tf_group_df['standardisation_function'].iloc[0]
        transform_function = std_tf_group_df['transform_function'].iloc[0]

        avg_metric_per_models.append({
            'standardisation_function': standardisation_function,
            'transform_function': transform_function,
            'avg_rmse_test': avg_rmse_test,
            'avg_r2_test': avg_r2_test,
        })

    return pd.DataFrame(avg_metric_per_models)

def main():
    df = get_results_df()

    best_model_dfs = pd.DataFrame()

    df = df.drop(columns=[
        'created_at',
        'train_dataset',
        'train_dataset_filename',
        'test_dataset',
        'test_dataset_filename',
        'r2_train_error',
        'rmse_train_error'
    ])

    for group_key, group_df in df.groupby([
        'model_type',
        'int_or_ext',
        'train_example_count',
        'test_example_count',
        'metric_of_interest',

    ]):

        models_df = calculate_average_metrics_from_models(group_df)
        if models_df is None:
            return None
        models_df.sort_values('avg_rmse_test', inplace=True)

        best_std = models_df['standardisation_function'].head(1).to_string()[5:]
        best_tf = models_df['transform_function'].head(1).to_string()[5:]
        
        best_std_df = group_df.loc[group_df['standardisation_function'] == best_std] 
        best_tf_df = best_std_df.loc[best_std_df['transform_function'] == best_tf]

        best_model_dfs = pd.concat([best_model_dfs, best_tf_df])

if __name__ == "__main__":
    if pytest.main(["-q", "./", "--exitfirst", "--ignore", "archive"]) == 0:
        main()
    else:
        logger.error("Tests failed.")
        sys.exit(1)
