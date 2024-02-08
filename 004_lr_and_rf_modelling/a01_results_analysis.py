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

def main():
    logging.basicConfig(level=logging.WARNING)
    df = get_model_results()

    print(df.head())

if __name__ == "__main__":
    main()