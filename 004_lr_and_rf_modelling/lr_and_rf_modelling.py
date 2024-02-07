from pprint import pprint
import json
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score, median_absolute_error, explained_variance_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
import pandas as pd
import numpy as np
from icecream import ic
from loguru import logger
from rich.console import Console
from rich.progress import track
from rich.markdown import Markdown
import os
import warnings
import sys
warnings.filterwarnings("ignore", category=RuntimeWarning)

console = Console()

#==========================================================================================================================
#                                                      CONSTANTS
#==========================================================================================================================

MODEL_TYPES = ["Linear Regression", "Random Forests"]
TEST_TYPES = ['interpolation', 'extrapolation']

TRAIN_DATASET_PATH = "./../000_datasets/2023-09-30_train_dataset.csv"
EXTRAPOLATION_TEST_DATASET_PATH = "./../000_datasets/2023-09-30_extrapolation_test_dataset.csv"

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
    
    'total_lost_samples',
    'avg_lost_samples',
    
    'total_lost_samples_percentage',
    'avg_lost_samples_percentage',
    
    'total_received_samples',
    'avg_received_samples',
    
    'total_received_samples_percentage',
    'avg_received_samples_percentage',
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

#==========================================================================================================================
#                                                      FUNCTIONS
#==========================================================================================================================

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

def get_error_for_output_variable(target_index, y_train, y_test, y_pred_train, y_pred_test, output_variable, error_type, transform_function):

    if y_train is None or y_test is None or y_pred_train is None or y_pred_test is None:
        logger.warning("Missing arguments")
        return None
    
    if output_variable is None:
        logger.warning("No output variable provided")
        return None
    
    if error_type is None:
        logger.warning("No error type provided")
        return None
    
    valid_error_types = [
        "rmse", "mse", "mae", "mape", "r2", "medae", "explained_variance"
    ]

    if error_type not in valid_error_types:
        logger.warning(f"Invalid error type: {error_type}")
        return None

    transformed_y_train_target = y_train[output_variable]
    transformed_y_test_target = y_test[output_variable]

    # ? Detransform all values before error metric calculation
    y_train_target = [ detransform_value(value, transform_function) for value in transformed_y_train_target ]
    y_test_target = [ detransform_value(value, transform_function) for value in transformed_y_test_target ]

    transformed_y_pred_train_target = y_pred_train[:, target_index]
    transformed_y_pred_test_target = y_pred_test[:, target_index]

    # ? Detransform all values before error metric calculation
    y_pred_train_target = [ detransform_value(value, transform_function) for value in transformed_y_pred_train_target ]
    y_pred_test_target = [ detransform_value(value, transform_function) for value in transformed_y_pred_test_target ]

    error_train = None
    error_test = None

    if error_type == "mae":
        error_train = mean_absolute_error(y_train_target, y_pred_train_target)
        error_test = mean_absolute_error(y_test_target, y_pred_test_target)

    elif error_type == "mse":
        error_train = mean_squared_error(y_train_target, y_pred_train_target)
        error_test = mean_squared_error(y_test_target, y_pred_test_target)   
    
    elif error_type == "rmse":
        try:
            error_train = mean_squared_error(y_train_target, y_pred_train_target, squared=False)
            error_test = mean_squared_error(y_test_target, y_pred_test_target, squared=False)
        except ValueError as e:
            logger.error(f"Error calculating RMSE: {e}")
            return None

    elif error_type == "mape":
        error_train = np.mean(
            np.abs([(y_train_target[i] - y_pred_train_target[i]) / y_train_target[i] * 100 for i in range(len(y_train_target))])
        )
        error_test = np.mean(
            np.abs([(y_test_target[i] - y_pred_test_target[i]) / y_test_target[i] * 100 for i in range(len(y_test_target))])
        )

    elif error_type == "r2":
        error_train = r2_score(y_train_target, y_pred_train_target)
        error_test = r2_score(y_test_target, y_pred_test_target)

    elif error_type == "medae":
        error_train = median_absolute_error(y_train_target, y_pred_train_target)
        error_test = median_absolute_error(y_test_target, y_pred_test_target)

    elif error_type == "explained_variance":
        error_train = explained_variance_score(y_train_target, y_pred_train_target)
        error_test = explained_variance_score(y_test_target, y_pred_test_target)

    else:
        logger.warning(f"Unknown error type: {error_type}")
        return None
    
    return error_train, error_test

def transform_df(df, output_variables, transform_function):
    if df is None:
        logger.warning("No dataframe provided")
        return None
    
    if output_variables is None:
        logger.warning("No output variables provided")
        return None
    
    if transform_function is None:
        logger.warning("No transform function provided")
        return None
    
    if transform_function not in TRANSFORM_FUNCTIONS:
        logger.warning(f"Unknown transform function: {transform_function}")
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
        logger.warning(f"Unknown transform function: {transform_function}")
        return None
    
    # ? Make sure df and transformed_df are different
    if df.equals(transformed_df) and transform_function != "none":
        logger.warning("df and transformed_df are the same after applying transformation")
        return None
    
    # ? Check if df and transformed_df have the same number of columns
    if len(df.columns) != len(transformed_df.columns):
        logger.warning("df and transformed_df have different number of columns")
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

def standardise_df(train_df, test_df, input_variables, standardisation_function):
    if train_df is None:
        logger.warning("No train dataframe provided")
        return None
    
    if test_df is None:
        logger.warning("No test dataframe provided")
        return None

    if input_variables is None:
        logger.warning("No input variables provided")
        return None
    
    if standardisation_function is None:
        logger.warning("No standardisation function provided")
        return None
    
    if standardisation_function not in STANDARDISATION_FUNCTIONS:
        logger.warning(f"Unknown standardisation function: {standardisation_function}")
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
        logger.warning(f"Unknown standardisation function: {standardisation_function}")
        return None

    # ? Make sure number of rows are the same before and after standardisation
    if len(train_df) != len(std_train_df) or len(test_df) != len(std_test_df):
        logger.warning("Number of rows are different before and after standardisation")
        return None

    # ? Make sure train_df and std_train_df are different
    if train_df.equals(std_train_df) and standardisation_function != "none":
        logger.warning("train_df and std_train_df are the same after applying standardisation")
        return None

    return std_train_df, std_test_df

def get_output_variables(metric):
    if metric is None:
        logger.warning("No metric provided")
        return []

    if metric not in METRICS:
        logger.warning(f"Unknown metric: {metric}")
        return []

    output_cols = []
    for STAT in STATS:
        output_cols.append(f"{metric}_{STAT}")

    if len(output_cols) != 19:
        logger.warning(f"Expected 19 output columns for {metric}, got {len(output_cols)}")
        return []

    return output_cols

def main():

    if TRAIN_DATASET_PATH is None:
        logger.error("No TRAIN_DATASET_PATH provided")
        return
    
    if EXTRAPOLATION_TEST_DATASET_PATH is None:
        logger.error("No EXTRAPOLATION_TEST_DATASET_PATH provided")
        return
    
    if not os.path.exists(TRAIN_DATASET_PATH):
        logger.error(f"TRAIN_DATASET_PATH does not exist: {TRAIN_DATASET_PATH}")
        return
    
    if not os.path.exists(EXTRAPOLATION_TEST_DATASET_PATH):
        logger.error(f"EXTRAPOLATION_TEST_DATASET_PATH does not exist: {EXTRAPOLATION_TEST_DATASET_PATH}")
        return
    
    EXTRAPOLATION_TEST_DF = pd.read_csv(EXTRAPOLATION_TEST_DATASET_PATH)
    MODEL_RESULT_DF = pd.DataFrame()

    for MODEL_TYPE in MODEL_TYPES:

        for TEST_TYPE in TEST_TYPES:

            if TEST_TYPE == "interpolation":
                df = pd.read_csv(TRAIN_DATASET_PATH)
                
                extrapolation_test_count = len(EXTRAPOLATION_TEST_DF)
                extrapolation_test_percentage = extrapolation_test_count / len(df) * 100

                # ? Split df into train and test where test has same number of rows as extrapolation test dataset
                # ? So both interpolation and extrapolation test datasets have the same number of rows
                # ? And both interpolation and extrapolation train datasets have the same number of rows
                TEST_DF = df.sample(frac=extrapolation_test_percentage/100, random_state=1)
                TRAIN_DF = df.drop(TEST_DF.index)

                test_dataset = "PCG + RCG (inclusive)"

            elif TEST_TYPE == "extrapolation":

                # ? Use the extrapolation test dataset as the test dataset
                TEST_DF = EXTRAPOLATION_TEST_DF
                test_count_percentage = len(TEST_DF) / len(TRAIN_DF) * 100

                TRAIN_DF = pd.read_csv(TRAIN_DATASET_PATH)
                TRAIN_DF = TRAIN_DF.sample(frac=1-(test_count_percentage/100), random_state=1)

                test_dataset = "PCG + RCG (exclusive)"

            else:
                logger.error(f"Unknown TEST_TYPE: {TEST_TYPE}")
                return

            if len(TRAIN_DF) == 0:
                logger.warning("No data in TRAIN_DF")
                return
            
            if len(TEST_DF) == 0:
                logger.warning("No data in TEST_DF")
                return
            
            if len(TRAIN_DF.columns) == 0:
                logger.warning("No columns in TRAIN_DF")
                return
            
            if len(TEST_DF.columns) == 0:
                logger.warning("No columns in TEST_DF")
                return

            if len(INPUT_VARIABLES) == 0:
                logger.warning("No input variables")
                return
            
            if len(TRAIN_DF.columns) != len(TEST_DF.columns):
                logger.warning("TRAIN_DF and TEST_DF have different number of columns")
                return

            for METRIC in track(METRICS, description=f"Modelling {MODEL_TYPE} {TEST_TYPE.capitalize()}..."):
                
                metric_index = METRICS.index(METRIC) + 1

                # console.print(f"[{metric_index}/{len(METRICS)}] Modelling {TEST_TYPE} {METRIC}...")
                
                if METRIC not in METRICS:
                    logger.warning(f"Unknown metric: {METRIC}")
                    continue

                output_variables = get_output_variables(METRIC)

                if len(output_variables) == 0:
                    logger.warning(f"No output variables for metric: {METRIC}")
                    continue

                for STANDARDISATION_FUNCTION in STANDARDISATION_FUNCTIONS:

                    for TRANSFORM_FUNCTION in TRANSFORM_FUNCTIONS:

                        # ? Cut df down to input and output variables
                        train_df = TRAIN_DF[INPUT_VARIABLES + output_variables]
                        test_df = TEST_DF[INPUT_VARIABLES + output_variables]

                        # ? Standardise input variables
                        std_train_df, std_test_df = standardise_df(train_df, test_df, INPUT_VARIABLES, STANDARDISATION_FUNCTION)

                        if std_train_df is None or std_test_df is None:
                            logger.warning("std_train_df or std_test_df is None")
                            continue

                        # ? Transform output variables
                        transformed_train_df = transform_df(std_train_df, output_variables, TRANSFORM_FUNCTION)
                        transformed_test_df = transform_df(std_test_df, output_variables, TRANSFORM_FUNCTION)

                        # ? Shuffle the data
                        transformed_train_df = transformed_train_df.sample(frac=1).reset_index(drop=True)
                        transformed_test_df = transformed_test_df.sample(frac=1).reset_index(drop=True)

                        X_train = transformed_train_df[INPUT_VARIABLES]
                        y_train = transformed_train_df[output_variables]

                        X_test = transformed_test_df[INPUT_VARIABLES]
                        y_test = transformed_test_df[output_variables]

                        if MODEL_TYPE == "Random Forests":
                            model = RandomForestRegressor()
                        else:
                            model = LinearRegression()
                        
                        model.fit(X_train, y_train)

                        y_pred_train = model.predict(X_train)
                        y_pred_test = model.predict(X_test)

                        created_at = pd.to_datetime('now').strftime("%Y-%m-%d %H:%M:%S")

                        created_at_path_format = created_at.replace(" ", "_").replace(":", "-")
                        
                        error_types = [
                            "rmse", "mse", "mae", "mape", "r2", "medae", "explained_variance"
                        ]

                        for output_variable in output_variables:
                            for error_type in error_types:
                                train_error, test_error = get_error_for_output_variable(
                                    target_index=output_variables.index(output_variable),
                                    y_train=y_train,
                                    y_test=y_test,
                                    y_pred_train=y_pred_train,
                                    y_pred_test=y_pred_test,
                                    output_variable=output_variable,
                                    error_type=error_type,
                                    transform_function=TRANSFORM_FUNCTION,
                                )
                                
                                input_variables_string = ", ".join(INPUT_VARIABLES)

                                model_result = {
                                    "model_type": MODEL_TYPE,
                                    "created_at": created_at,
                                    "int_or_ext": TEST_TYPE,
                                    "train_dataset": "PCG + RCG (inclusive)",
                                    "train_dataset_filename": TRAIN_DATASET_PATH.split("/")[-1],
                                    "test_dataset": test_dataset,
                                    "test_dataset_filename": EXTRAPOLATION_TEST_DATASET_PATH.split("/")[-1],
                                    "train_example_count": len(transformed_train_df),
                                    "test_example_count": len(transformed_test_df),
                                    "input_variables": input_variables_string,
                                    "output_variable": output_variable,
                                    "metric_of_interest": METRIC,
                                    "standardisation_function": STANDARDISATION_FUNCTION,
                                    "transform_function": TRANSFORM_FUNCTION,
                                    "error_type": error_type,
                                    "train_value": train_error,
                                    "test_value": test_error,
                                }

                                MODEL_RESULT_DF = pd.concat([MODEL_RESULT_DF, pd.DataFrame([model_result])], ignore_index=True)

                        # ? Save model in joblib  
                        model_filename = f"all_models/{MODEL_TYPE.replace(' ', '_')}_models/{METRIC}/{created_at_path_format}_{TEST_TYPE}_{STANDARDISATION_FUNCTION}_{TRANSFORM_FUNCTION}.joblib"
                        os.makedirs(os.path.dirname(model_filename), exist_ok=True)
                        joblib.dump(model, model_filename)

    MODEL_RESULT_DF.to_csv(f"{created_at_path_format}_results.csv", index=False)

main()