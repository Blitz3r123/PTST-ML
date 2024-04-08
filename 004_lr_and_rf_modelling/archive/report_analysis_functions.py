#===========================================================================
#  ?                                ABOUT
#  @author         :  Kaleem Peeroo
#  @email          :  KaleemPeeroo@gmail.com
#  @repo           :  
#  @createdOn      :  2024-01-30
#  @description    :  Functions to collect all results from all LR and RF models into a single dataframe.
#===========================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json
import re

from pprint import pprint

#==========================================================================
#                              Constants
#==========================================================================

LR_DIR = "./lr_models"
RF_DIR = "./rf_models"

METRICS = [
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

CUSTOM_ORDER = {'mean': -4, 'std': -3, 'min': -2, 'max': -1}
REVERSE_CUSTOM_ORDER = {v: k for k, v in CUSTOM_ORDER.items()}

#==========================================================================
#                              Functions
#==========================================================================

def validate_final_df(final_df):
    if len(final_df) == 0:
        print("Final dataframe is empty")

    if 'model_type' not in final_df.columns or 'int_or_ext' not in final_df.columns:
        print("Expected 'model_type' and 'int_or_ext' columns in final dataframe")
        return

    grouped_final_df = final_df.groupby(['model_type', 'int_or_ext'])

    for (model_type, int_or_ext), data in grouped_final_df:

        if len(data) == 0:
            print(f"{model_type.capitalize()} {int_or_ext.capitalize()} dataframe is empty")
            continue

        if 'dds_metric' not in data.columns:
            print(f"{model_type.capitalize()} {int_or_ext.capitalize()} dataframe does not have 'dds_metric' column")
            continue

        if data['dds_metric'].nunique() != len(METRICS):
            missing = set(METRICS) - set(data['dds_metric'].unique())

            print(f"{model_type.capitalize()} {int_or_ext.capitalize()} with {len(data)} rows does not have all metrics.")
            print(f"Expected {len(METRICS)} unique values for 'dds_metric' column, got {data['dds_metric'].nunique()}")
            print(f"Missing:")
            for item in missing:
                print(f"\t- {item}")
            print()

def format_with_commas(x):
    return '{:,.3f}'.format(x)

def format_non_percentile_stat(x):
    if "mean" in x:
        return "Mean"
    elif "std" in x:
        return "Standard Deviation"
    elif "min" in x:
        return "Minimum"
    elif "max" in x:
        return "Maximum"
    else:
        return x

def format_number_suffix(x):
    if (x-1) % 10 == 0:
        return f"{x}st"
    elif (x-2) % 10 == 0:
        return f"{x}nd"
    elif (x-3) % 10 == 0:
        return f"{x}rd"
    else:
        return f"{x}th"

def add_borders_to_tex_table(tex_content, error_metric):
    re_borders = re.compile(r"begin\{tabular\}\{([^\}]+)\}")
    borders = re_borders.findall(tex_content)[0]
    borders = '|'.join(list(borders))
    borders = '|' + borders + '|'
    tex_content = re_borders.sub("begin{tabular}{" + borders + "}", tex_content)

    tex_content = tex_content.replace("\\\\", "\\\\\n\\hline")
    tex_content = tex_content.replace("\\toprule", "\\hline")
    tex_content = tex_content.replace("\\midrule", "\\hline")
    tex_content = tex_content.replace("\\bottomrule", "\\hline")
    tex_content = tex_content.replace("\\hline\n\\hline", "\\hline")
    tex_content = tex_content.replace("[h]", "[h]\n\\centering")

    tex_content = tex_content.replace("Value", error_metric.upper())

    return tex_content

# Function to convert values to sortable keys
def convert_to_sortable(x):

    if x in CUSTOM_ORDER:
        return CUSTOM_ORDER[x]
        
    try:
        # Convert numeric strings to integers
        return int(x)
    except ValueError:
        # Keep non-numeric strings as is
        return float('inf')
    
def is_dir_empty(dirpath):
    return len(os.listdir(dirpath)) == 0

def model_dir_has_metrics(dirpath):
    return set(METRICS).issubset(set(os.listdir(dirpath)))

def validate_model_dir(model_dir):
    if not os.path.exists(model_dir):
        print(f"Directory {model_dir} does not exist")

    if is_dir_empty(model_dir):
        print(f"Directory {model_dir} is empty")

    if not model_dir_has_metrics(model_dir):
        print(f"Directory {model_dir} does not have all metrics")

def get_model_results():
    validate_model_dir(LR_DIR)
    validate_model_dir(RF_DIR)

    lr_models = [os.path.join(LR_DIR, model) for model in os.listdir(LR_DIR)]
    lr_models = [model for model in lr_models if os.path.isdir(model)]

    if len(lr_models) != len(METRICS):
        print(f"Expected {len(METRICS)} LR models, got {len(lr_models)}")

    rf_models = [os.path.join(RF_DIR, model) for model in os.listdir(RF_DIR)]
    rf_models = [model for model in rf_models if os.path.isdir(model)]

    if len(rf_models) != len(METRICS):
        print(f"Expected {len(METRICS)} RF models, got {len(rf_models)}")

    final_df = pd.DataFrame()

    all_models = lr_models + rf_models

    for model in all_models:
        
        model_results = [file for file in os.listdir(model) if file.endswith(".json")]
        model_results = [os.path.join(model, result) for result in model_results]
        
        if len(model_results) == 0:
            print(f"Model {model} has no json files")
            continue

        test_file = model_results[0]
        with open(test_file, "r") as f:
            data = json.load(f)

        if len(data) == 0:
            print(f"Model {model} has empty json files")
            continue

        filename = os.path.basename(test_file)

        required_keys = [
            'model_type',
            'metric_of_interest',
            'errors_per_output_variable',
            'standardisation_function',
            'transform_function',
            'train_dataset',
            'test_dataset'
        ]

        if not set(required_keys).issubset(set(data.keys())):
            print(f"Model {model} does not have all required keys.")
            print(f"Required keys: {required_keys}")
            print(f"Model keys: {list(data.keys())}")
            continue

        # Reduce the data to the required_keys
        data = {key: data[key] for key in required_keys}

        required_error_metrics = ['r2', 'rmse', 'mae', 'mape', 'medae']

        if not set(required_error_metrics).issubset(set(data['errors_per_output_variable'].keys())):
            print(f"Model {model} does not have all required error metrics.")
            print(f"Required error metrics: {required_error_metrics}")
            print(f"Model error metrics: {list(data['errors_per_output_variable'].keys())}")
            continue

        # Reduce the data to the required_error_metrics
        data['errors_per_output_variable'] = {key: data['errors_per_output_variable'][key] for key in required_error_metrics}

        for error_metric in required_error_metrics:
            data[error_metric] = data['errors_per_output_variable'][error_metric]
            del data['errors_per_output_variable'][error_metric]

        if len(data) == 0:
            print(f"Model {model} has no data after reduction")
            continue

        model_type = data['model_type']
        standardisation_function = data['standardisation_function']
        transform_function = data['transform_function']

        for error_metric in required_error_metrics:
            error_metric_data = data[error_metric]
            
            if len(error_metric_data) == 0:
                print(f"Model {model} has no data for {error_metric}")
                continue

            if len(error_metric_data) != 19:
                print(f"Model {model} has {len(error_metric_data)} items for {error_metric}, expected 19")
                continue

            for item in error_metric_data:

                if len(item.keys()) != 3:
                    print(f"Model {model} {item} does NOT have 3 items.")
                    continue

                row_data = {}

                train_keys = [key for key in item.keys() if key.endswith("train")]
                if len(train_keys) == 0:
                    print(f"Model {model} has no train keys")
                    continue

                test_keys = [key for key in item.keys() if key.endswith("test")]
                if len(test_keys) == 0:
                    print(f"Model {model} has no test keys")
                    continue
                
                train_value = item[train_keys[0]]
                test_value = item[test_keys[0]]
                output_variable = item['output_variable']

                for type in ['train', 'test']:
                    row_data['train_or_test'] = type
                    row_data['model_type'] = model_type
                    row_data['standardisation_function'] = standardisation_function
                    row_data['transform_function'] = transform_function
                    row_data['value'] = train_value if type == 'train' else test_value
                    row_data['output_variable'] = output_variable
                    row_data['error_metric'] = error_metric
                    row_data['filename'] = filename
                    row_data['train_dataset'] = data['train_dataset']
                    row_data['test_dataset'] = data['test_dataset']

                    int_or_ext = filename.replace(".json", "").split("_")[2]
                    row_data['int_or_ext'] = int_or_ext

                    final_df = pd.concat([final_df, pd.DataFrame([row_data])])

    test_train_expected_row_count = len(all_models) * len(required_error_metrics) * 19 * 2
    if final_df.shape[0] != test_train_expected_row_count:
        print(f"Expected {test_train_expected_row_count} rows, got {final_df.shape[0]}")

    # Create 'train_value' and 'test_value' columns based on 'train_or_test' column
    final_df['train_value'] = final_df.apply(lambda x: x['value'] if x['train_or_test'] == 'train' else None, axis=1)
    final_df['test_value'] = final_df.apply(lambda x: x['value'] if x['train_or_test'] == 'test' else None, axis=1)
    
    # Drop the 'train_or_test' and 'value' columns
    final_df.drop(columns=['train_or_test', 'value'], inplace=True)
    
    # Isolate the train and test data and pick the first value
    final_df = final_df.groupby([
            'model_type', 
            'standardisation_function', 
            'transform_function', 
            'output_variable', 
            'error_metric', 
            'filename', 
            'train_dataset', 
            'test_dataset', 
            'int_or_ext'
    ]).agg({
        'train_value': 'first',
        'test_value': 'first'
    }).reset_index()
    
    final_df['dds_metric'] = final_df['output_variable'].apply(
        lambda x: "_".join(x.split("_")[:-1])
    )
    
    if final_df['dds_metric'].nunique() != len(METRICS):
        print(f"Expected {len(METRICS)} unique values for 'dds_metric' column, got {final_df['dds_metric'].nunique()}")

    validate_final_df(final_df)

    if final_df['int_or_ext'].nunique() != 2:
        print(final_df['int_or_ext'].nunique())
        print(final_df['int_or_ext'].unique())
        print(f"Expected 2 unique values for 'int_or_ext' column, got {final_df['int_or_ext'].nunique()}")

    if final_df['model_type'].nunique() != 2:
        print(final_df['model_type'].nunique())
        print(final_df['model_type'].unique())
        print(f"Expected 2 unique values for 'model_type' column, got {final_df['model_type'].nunique()}")


    return final_df