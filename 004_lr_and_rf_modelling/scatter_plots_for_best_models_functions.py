import os
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.model_selection import train_test_split
import joblib
import warnings
import json
from icecream import ic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=RuntimeWarning)

ERROR_TYPES = [
    "rmse", "mse", "mae", "mape", "r2", "medae", "explained_variance"
]

STANDARDISATION_FUNCTIONS = [
    "none",
    "z_score",
    'min_max',
    'robust_scaler',
]

INPUT_VARIABLES = [
    'datalen_bytes',
    'pub_count',
    'sub_count', 
    'reliability',
    'multicast',
    'durability_0',
    'durability_1',
    'durability_2',
    'durability_3',
]

METRICS = [
    'latency_us',
    'total_throughput_mbps',
    'avg_throughput_mbps',
    'total_samples_per_sec',
    'avg_samples_per_sec',
    # 'total_lost_samples',
    # 'avg_lost_samples',
    # 'total_lost_samples_percentage',
    'avg_lost_samples_percentage',
    # 'total_received_samples',
    # 'avg_received_samples',
    # 'total_received_samples_percentage',
    'avg_received_samples_percentage',
]

TRANSFORM_FUNCTIONS = [
    'none',
    'log',
    'log10',
    'log1p',
    'log2',
    'sqrt',
]

ALL_BEST_MODELS_FILEPATH = "./all_best_models.csv"
ALL_BEST_MODELS_DF = pd.read_csv(ALL_BEST_MODELS_FILEPATH)

FULL_DF = pd.read_csv("./../000_datasets/2023-09-30_20_percent_truncated_dataset.csv")

def get_model_json_from_filename(filename):
    if filename is None:
        return None
    
    if not os.path.exists(filename):
        print(f"Model file {filename} does not exist")
        return None
    
    with open(filename, 'r') as f:
        return json.load(f)
    
def get_avg_train_test_error(errors_per_target, error_type):
    if errors_per_target is None or error_type is None:
        print("errors_per_target or error_type is None")
        return None, None
    if len(errors_per_target) == 0:
        print("errors_per_target is empty")
        return None, None
    if f'{error_type}_train' not in errors_per_target[0]:
        print(f"{error_type}_train not in errors_per_target")
        return None, None
    if f'{error_type}_test' not in errors_per_target[0]:
        print("rmse_test not in errors_per_target")
        return None, None

    all_train_errors = [target[f'{error_type}_train'] for target in errors_per_target]
    all_test_errors = [target[f'{error_type}_test'] for target in errors_per_target]

    if len(all_train_errors) == 0:
        print("all_train_errors is empty")
        return None, None
    if len(all_test_errors) == 0:
        print("all_test_errors is empty")
        return None, None
    
    avg_train_error = np.mean(all_train_errors)
    avg_test_error = np.mean(all_test_errors)

    return avg_train_error, avg_test_error

def standardise_df(train_df, test_df, input_variables, standardisation_function):
    if train_df is None:
        print("No train dataframe provided")
        return None
    
    if test_df is None:
        print("No test dataframe provided")
        return None

    if input_variables is None:
        print("No input variables provided")
        return None
    
    if standardisation_function is None:
        print("No standardisation function provided")
        return None
    
    if standardisation_function not in STANDARDISATION_FUNCTIONS:
        print(f"Unknown standardisation function: {standardisation_function}")
        return None

    std_train_df = train_df.copy()
    std_test_df = test_df.copy()

    X_train = std_train_df[input_variables]
    X_test = std_test_df[input_variables]

    if standardisation_function == "none":
        pass

    elif standardisation_function == "z_score":
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        std_train_df[input_variables] = X_train_scaled
        std_test_df[input_variables] = X_test_scaled

    elif standardisation_function == "min_max":
        scaler = MinMaxScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        std_train_df[input_variables] = X_train_scaled
        std_test_df[input_variables] = X_test_scaled

    elif standardisation_function == "robust_scaler":
        scaler = RobustScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        std_train_df[input_variables] = X_train_scaled
        std_test_df[input_variables] = X_test_scaled

    else:
        print(f"Unknown standardisation function: {standardisation_function}")
        return None

    # ? Make sure train_df and std_train_df are different
    if train_df.equals(std_train_df) and standardisation_function != "none":
        print("train_df and std_train_df are the same after applying standardisation")
        return None

    return std_train_df, std_test_df

def detransform_value(value, transform_function):
    if value is None or transform_function is None:
        print("Missing arguments")
        return None
    
    if transform_function not in TRANSFORM_FUNCTIONS:
        print(f"Invalid transform_function: {transform_function}")
        return None
    
    if transform_function == 'none':
        return value
    
    elif transform_function == 'log':
        return np.exp(value)
    
    elif transform_function == 'log10':
        return np.power(10, value)
    
    elif transform_function == 'log1p':
        return np.exp(value) - 1
    
    elif transform_function == 'log2':
        return np.power(2, value)
    
    elif transform_function == 'sqrt':
        return np.power(value, 2)
    
    else:
        print(f"Unknown transform_function: {transform_function}")
        return None
    
def transform_df(df, output_variables, transform_function):
    if df is None:
        print("No dataframe provided")
        return None
    
    if output_variables is None:
        print("No output variables provided")
        return None
    
    if transform_function is None:
        print("No transform function provided")
        return None
    
    if transform_function not in TRANSFORM_FUNCTIONS:
        print(f"Unknown transform function: {transform_function}")
        return None
    
    transformed_df = df.copy()

    if transform_function == "none":
        pass

    elif transform_function == "log":
        for output_variable in output_variables:
            transformed_df[output_variable] = transformed_df[output_variable].apply(lambda x: np.log(x))
    
    elif transform_function == "log10":
        for output_variable in output_variables:
            transformed_df[output_variable] = transformed_df[output_variable].apply(lambda x: np.log10(x))

    elif transform_function == "log2":
        for output_variable in output_variables:
            transformed_df[output_variable] = transformed_df[output_variable].apply(lambda x: np.log2(x))

    elif transform_function == "log1p":
        for output_variable in output_variables:
            transformed_df[output_variable] = transformed_df[output_variable].apply(lambda x: np.log1p(x))

    elif transform_function == "sqrt":
        for output_variable in output_variables:
            transformed_df[output_variable] = transformed_df[output_variable].apply(lambda x: np.sqrt(x))

    else:
        print(f"Unknown transform function: {transform_function}")
        return None
    
    # ? Make sure df and transformed_df are different
    if df.equals(transformed_df) and transform_function != "none":
        print("df and transformed_df are the same after applying transformation")
        return None
    
    # ? Check if df and transformed_df have the same number of columns
    if len(df.columns) != len(transformed_df.columns):
        print("df and transformed_df have different number of columns")
        return None
    
    # ? Check for NaNs
    if transformed_df.isnull().values.any():
        original_row_count = len(transformed_df)
        nan_count_before = transformed_df.isnull().values.sum()
        transformed_df = transformed_df.dropna()
        nan_count_after = transformed_df.isnull().values.sum()
        # print(f"Removed {nan_count_before - nan_count_after} NaNs from transformed_df. Originally {original_row_count} rows, now {len(transformed_df)} rows.")
    
    # ? Check for Infs
    if np.isinf(transformed_df).values.any():
        original_row_count = len(transformed_df)
        inf_count_before = np.isinf(transformed_df).values.sum()
        transformed_df = transformed_df.replace([np.inf, -np.inf], np.nan)
        transformed_df = transformed_df.dropna()
        inf_count_after = np.isinf(transformed_df).values.sum()
        # print(f"Removed {inf_count_before - inf_count_after} Infs from transformed_df. Originally {original_row_count} rows, now {len(transformed_df)} rows.")
    
    return transformed_df

def get_input_variable_names_from_values(values_list):
    if values_list is None:
        print("No values given")
        return
    
    if len(values_list) == 0:
        print("No values given")
        return
    
    if len(values_list) != len(INPUT_VARIABLES):
        print(f"Found {len(values_list)} values when there are {len(INPUT_VARIABLES)}")
        return
    
    input_variable_names = {}

    for INPUT_VARIABLE in INPUT_VARIABLES:
        element_index = INPUT_VARIABLES.index(INPUT_VARIABLE)
        input_variable_names[INPUT_VARIABLE] = values_list[element_index]

    return input_variable_names

def get_input_title_from_input_values(variable_names_and_values):
    if variable_names_and_values is None:
        print("No args given")
        return
    
    title = ""
    durability_key = [k for (k, v) in variable_names_and_values.items() if 'durability' in k and v == 1][0]
    durability_value = durability_key.split("_")[1]

    for key, value in variable_names_and_values.items():
        if key == "datalen_bytes":
            datalen_value = value
            if value >= 1000:
                value = int(value)
                value = int( value / 1000 )
                unit = "KB"
            else:
                unit = "B"

            title += f"{value}{unit} "
        
        elif key == "pub_count":
            title += f"{value}P "
        
        elif key == "sub_count":
            title += f"{value}S "
        
        elif key == "reliability":
            if value == 1:
                title += "REL "
            else:
                title += "BE "
        
        elif key == "multicast":
            if value == 0:
                title += "UC "
            else:
                title += "MC "

        else:
            pass

    title += f"{durability_value}DUR "

    return title

def get_best_model_for_metric(metric):
    if metric is None:
        print("Missing arguments")
        return None, None
    if metric not in METRICS:
        print(f"Unknown metric: {metric}")
        return None, None

    row = ALL_BEST_MODELS_DF[ALL_BEST_MODELS_DF['metric'] == metric]
    row = row.sort_values(by=['rmse_test', 'r2_test'], ascending=[True, False])

    if len(row) == 0:
        print(f"No best model found for metric {metric}")
        return None, None

    best_model = row.iloc[0]

    model_type = best_model['model_type']
    if model_type == 'random forests':
        model_type = 'rf'
    else:
        model_type = 'lr'

    # ? Get the model json
    model_json_filename = row['filename'].values[0]
    model_json_filepath = os.path.join(f"{model_type}_models", metric, model_json_filename)
    model_json = get_model_json_from_filename(model_json_filepath)
    if model_json is None:
        return None
    
    # ? Get model joblib
    model_joblib_filepath = model_json_filepath.replace(".json", ".joblib")
    if not os.path.exists(model_joblib_filepath):
        print(f"Model joblib file {model_joblib_filepath} does not exist")
        return None
    model = joblib.load(model_joblib_filepath)

    return model_json, model

def predict_random_test_for_metric(model_json, model, metric, test_count, plot_type):
    if 'model_name' not in model_json.keys():
        print("model_name not in model_json")
        return None
    model_name = model_json['model_name']

    required_keys = ['input_variables', 'output_variables', 'standardisation_function', 'transform_function']
    for key in required_keys:
        if key not in model_json.keys():
            print(f"Key {key} not found in model json")
            continue

    #  ? Get the average train and test rmse for that model
    avg_train_rmse, avg_test_rmse = get_avg_train_test_error(
        model_json['errors_per_output_variable']['rmse'],
        'rmse'
    )
    if avg_train_rmse is None or avg_test_rmse is None:
        return None

    #  ? Get the average train and test R2 for that model
    avg_train_r2, avg_test_r2 = get_avg_train_test_error(
        model_json['errors_per_output_variable']['r2'],
        'r2'
    )
    if avg_train_rmse is None or avg_test_rmse is None:
        return None

    standardisation_function = get_info_from_model_json(model_json, 'standardisation_function')
    transform_function = get_info_from_model_json(model_json, 'transform_function')

    input_variables = get_info_from_model_json(model_json, 'input_variables')
    output_variables = get_info_from_model_json(model_json, 'output_variables')

    # ? Split raw data into train and test
    X_test, X_train, y_test, y_train = train_test_split(FULL_DF[input_variables], FULL_DF[output_variables], test_size=0.5, random_state=42)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # ? Standardise train and test
    std_train_df, std_test_df = standardise_df(train_df, test_df, input_variables, standardisation_function)
    if std_train_df is None or std_test_df is None:
        return None

    # ? Transform output variables
    transformed_train_df = transform_df(std_train_df, output_variables, transform_function)
    transformed_test_df = transform_df(std_test_df, output_variables, transform_function)

    # ? Shuffle the data
    transformed_train_df = transformed_train_df.sample(frac=1).reset_index(drop=True)
    transformed_test_df = transformed_test_df.sample(frac=1).reset_index(drop=True)

    std_X_train = transformed_train_df[INPUT_VARIABLES]
    trans_y_train = transformed_train_df[output_variables]

    std_X_test = transformed_test_df[INPUT_VARIABLES]
    trans_y_test = transformed_test_df[output_variables]

    # ? Get the predictions
    trans_y_test_pred = model.predict(std_X_test)

    # ? Detransform the predictions and actual values
    y_test_pred = detransform_value(trans_y_test_pred, transform_function)
    y_test = detransform_value(trans_y_test, transform_function)

    # ? Plot the predictions against the actual values
    row_count = test_count
    fig, axs = plt.subplots(row_count, row_count, figsize=(20, 20))
    fig.suptitle(f'{model_name}')

    for i in range(row_count * row_count):
        row = i // row_count
        col = i % row_count

        ax = axs[row, col]

        # ? Generate a random index - pick a random test
        random_index = np.random.randint(0, len(y_test_pred))

        input_variable_names = get_input_variable_names_from_values(X_test.values[random_index])
        input_title = get_input_title_from_input_values(input_variable_names)

        actual_mean = np.mean(y_test.values[random_index])
        actual_mean = round(actual_mean, 2)

        predicted_mean = np.mean(y_test_pred[random_index])
        predicted_mean = round(predicted_mean, 2)

        # ? Get max value
        max_value = max(y_test.values[random_index].max().max(), y_test_pred[random_index].max().max())

        if plot_type == 'scatter':
            ax.scatter(
                y_test.values[random_index], 
                y_test_pred[random_index], 
                s=3, 
                label=f"Test RMSE: {avg_test_rmse:,.2f}\nTest R2: {avg_test_r2:.2f}"
            )
            ax.axline(
                [0, 0], 
                [1, 1], 
                color='#de425b', 
                alpha=0.2, 
                linestyle='--', 
                label=f"Std: {standardisation_function}\nTrans: {transform_function}"
            )
            ax.set_xlim(0, max_value * 1.1)
            ax.set_ylim(0, max_value * 1.1)
            
        elif plot_type == "cdf":
            ax.hist(
                y_test.values[random_index], 
                bins=200, 
                density=True, 
                histtype='step', 
                cumulative=True, 
                label=f'Actual', 
                color='#488f31'
            )
            ax.hist(
                y_test_pred[random_index], 
                bins=200, 
                density=True, 
                histtype='step', 
                cumulative=True, 
                label=f'Predicted', 
                color='#de425b'
            )
            ax.plot([], label=f"\nTest RMSE: \n{avg_test_rmse:,.2f}\nTest R2: \n{avg_test_r2:.2f}", color="#fff", alpha=0)

        ax.set_title(input_title)
        ax.set_xlabel(f'{metric}')
        ax.set_ylabel(f'F(x)')
        ax.legend()

    plt.tight_layout()
    plt.show()

def get_info_from_model_json(model_json, key):
    info = model_json[key]

    if info is None:
        print(f"{key} is None")
        return None

    if key == "standardisation_function":
        if info not in STANDARDISATION_FUNCTIONS:
            print(f"Unknown standardisation_function: {info}")
            return None
    
    elif key == "transform_function":
        if info not in TRANSFORM_FUNCTIONS:
            print(f"Unknown transform_function: {info}")            
            return None
    
    else:
        pass

    return info

def predict_random_tests_for_metric_with_gridspace_location(model_json, model, metric, test_count):
    if model_json is None or model is None or metric is None or test_count is None:
        print("Missing arguments")
        return None
    
    if 'model_name' not in model_json.keys():
        print("model_name not in model_json")
        return None
    
    required_keys = ['input_variables', 'output_variables', 'standardisation_function', 'transform_function']
    for key in required_keys:
        if key not in model_json.keys():
            print(f"Key {key} not found in model json")
            continue

    #  ? Get the average train and test rmse for that model
    avg_train_rmse, avg_test_rmse = get_avg_train_test_error(
        model_json['errors_per_output_variable']['rmse'],
        'rmse'
    )
    if avg_train_rmse is None or avg_test_rmse is None:
        return None
    
    #  ? Get the average train and test R2 for that model
    avg_train_r2, avg_test_r2 = get_avg_train_test_error(
        model_json['errors_per_output_variable']['r2'],
        'r2'
    )


    standardisation_function = get_info_from_model_json(model_json, 'standardisation_function')
    transform_function = get_info_from_model_json(model_json, 'transform_function')

    # ? Detransform the values
    avg_train_rmse = detransform_value(avg_train_rmse, transform_function)
    avg_test_rmse = detransform_value(avg_test_rmse, transform_function)

    input_variables = get_info_from_model_json(model_json, 'input_variables')
    output_variables = get_info_from_model_json(model_json, 'output_variables')

    # ? Split raw data into train and test
    X_test, X_train, y_test, y_train = train_test_split(FULL_DF[input_variables], FULL_DF[output_variables], test_size=0.5, random_state=42)

    actual_inputs, actual_outputs, predicted_outputs = predict(
        model, 
        input_variables,
        output_variables,
        X_train, 
        y_train, 
        X_test, 
        y_test, 
        standardisation_function, 
        transform_function
    )

    actual_inputs.reset_index(drop=True, inplace=True)
    actual_outputs.reset_index(drop=True, inplace=True)
    
    # ? Select test_count random rows
    random_rows = np.random.choice(actual_inputs.index, size=test_count, replace=False)
    
    # ? Get the actual inputs, outputs, and predicted outputs for the random rows
    random_actual_inputs = actual_inputs.loc[random_rows]
    random_actual_outputs = actual_outputs.loc[random_rows]
    random_predicted_outputs = predicted_outputs[random_rows]

    fig, axs = plt.subplots(test_count, 2, figsize=(8, 4 * test_count))
    for i in range(test_count):
        inputs = random_actual_inputs.iloc[i].values
        input_variable_names = get_input_variable_names_from_values(inputs)
        input_title = get_input_title_from_input_values(input_variable_names)

        outputs = random_actual_outputs.iloc[i].values
        predicted_outputs = random_predicted_outputs[i]

        max_value = max(outputs.max(), predicted_outputs.max())

        row = i % test_count

        ax = axs[row, 0]
        ax.scatter(FULL_DF['pub_count'], FULL_DF['datalen_bytes'] / 1000, s=3)
        ax.scatter(inputs[1], int(inputs[0] / 1000), s=100, color='#ffa600', label=f"{inputs[1]}P\n{ int(inputs[0] / 1000) }KB\n{inputs[2]}S")
        ax.set_title(input_title)
        ax.set_xlabel('pub_count')
        ax.set_ylabel('Data Length (KB)')
        ax.legend()

        pub_count = 10
        sub_count = 25
        datalen_kb = 80

        min_pub_count = FULL_DF['pub_count'].min()
        max_pub_count = FULL_DF['pub_count'].max()
        min_sub_count = FULL_DF['sub_count'].min()
        max_sub_count = FULL_DF['sub_count'].max()
        min_datalen = FULL_DF['datalen_bytes'].min()
        max_datalen = FULL_DF['datalen_bytes'].max()

        ax.axvspan(
            0, 
            pub_count, 
            ymin=0, 
            ymax=(datalen_kb * 1000) / max_datalen, 
            color='#488f31', 
            alpha=0.1, 
        )
        ax.axvspan(
            pub_count,
            max_pub_count,
            ymin=(datalen_kb * 1000) / max_datalen, 
            ymax=max_datalen, 
            color='#003f5c', 
            alpha=0.1, 
        )
        ax.axvspan(
            0,
            pub_count,
            ymin=(datalen_kb * 1000) / max_datalen, 
            ymax=max_datalen, 
            color='#003f5c', 
            alpha=0.1,
        )
        ax.axvspan(
            pub_count,
            max_pub_count,
            ymin=0,
            ymax=(datalen_kb * 1000) / max_datalen,
            color='#003f5c',
            alpha=0.1,
        )
        ax.set_xlim(min_pub_count, max_pub_count)
        ax.set_ylim(min_datalen / 1000, max_datalen / 1000)

        ax = axs[row, 1]
        ax.scatter(outputs, predicted_outputs, s=3, label=f"Test RMSE: {avg_test_rmse:.2f}\nTest R2: {avg_test_r2:.2f}")
        ax.set_title(input_title)
        ax.set_xlabel(f'Actual {metric}')
        ax.set_ylabel(f'Predicted {metric}')
        ax.axline(
            [0, 0], 
            [1, 1], 
            color='#de425b', 
            alpha=0.2, 
            linestyle='--', 
            label=f"Std: {standardisation_function}\nTrans: {transform_function}"
        )
        ax.set_xlim(0, max_value * 1.1)
        ax.set_ylim(0, max_value * 1.1)
        ax.legend()
    
    plt.tight_layout()
    plt.show()

def predict(
    model, 
    input_variables, 
    output_variables, 
    X_train, 
    y_train, 
    X_test, 
    y_test, 
    std_function, 
    trans_function, 
    ):
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # ? Standardise train and test
    std_train_df, std_test_df = standardise_df(train_df, test_df, input_variables, std_function)
    if std_train_df is None or std_test_df is None:
        return None
    
    # ? Transform output variables
    transformed_train_df = transform_df(std_train_df, output_variables, trans_function)
    transformed_test_df = transform_df(std_test_df, output_variables, trans_function)

    # ? Shuffle the data
    transformed_train_df = transformed_train_df.sample(frac=1).reset_index(drop=True)
    transformed_test_df = transformed_test_df.sample(frac=1).reset_index(drop=True)

    # std_X_train = transformed_train_df[INPUT_VARIABLES]
    # trans_y_train = transformed_train_df[output_variables]

    std_X_test = transformed_test_df[INPUT_VARIABLES]
    trans_y_test = transformed_test_df[output_variables]

    # ? Get the predictions
    trans_y_test_pred = model.predict(std_X_test)

    # ? Detransform the predictions and actual values
    y_test_pred = detransform_value(trans_y_test_pred, trans_function)
    y_test = detransform_value(trans_y_test, trans_function)

    # ? Cut X_test to have same rows as y_test_pred
    if X_test.shape[0] > y_test_pred.shape[0]:
        X_test = X_test[:y_test_pred.shape[0]]

    return X_test, y_test, y_test_pred