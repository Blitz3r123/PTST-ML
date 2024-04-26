#==========================================================================================================================
#  ?                                                     ABOUT
#  @author         :    Kaleem Peeroo
#  @email          :    Kaleem.Peeroo@city.ac.uk 
#  @repo           :    https://github.com/Blitz3r123/PTST-ML 
#  @createdOn      :    2024-02-20 
#  @description    :    Take raw test data and put into a single spreadsheet. 
#==========================================================================================================================

import os
import sys
import pytest
import logging
import pandas as pd
from icecream import ic

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#--------------------------------------------------------------------------------------------------------------------------
#                                                      FUNCTIONS
#--------------------------------------------------------------------------------------------------------------------------
def get_longest_path_in_dir(dir_path: str) -> str:
    """
    Get the longest path in a directory.
    """
    longest_path = ""
    longest_path_len = 0

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            if len(file_path) > longest_path_len:
                longest_path = file_path
                longest_path_len = len(file_path)
    
    return longest_path

def get_test_parent_dirpath_from_fullpath(longest_path: str = "") -> str:
    if "/" not in longest_path:
        logger.error(f"No / found in {longest_path}.")
        return None

    longest_path_items = longest_path.split("/")
    if len(longest_path_items) <= 2:
        return longest_path

    return "/".join(longest_path_items[:-2])

def get_file_line_count(file_path):
    with open(file_path, "r") as f:
        num_lines = sum(1 for line in f)

    return num_lines

def get_latency_df_from_testdir(test_dir: str) -> pd.DataFrame:
    logger.info(f"Getting latency df from {test_dir}.")
    pub_file = get_pub_file_from_testdir(test_dir)
    if pub_file is None:
        return None

    file_line_count = get_file_line_count(pub_file)
    if file_line_count <= 5:
        logger.warning(f"Only {file_line_count} lines found in {pub_file}.")
        return None

    logger.info(f"Reading first 5 lines of {pub_file}.")
    # ? Read the first 5 lines of the file
    with open(pub_file, "r") as f:
        head = [next(f) for x in range(5)]

    logger.info(f"Looking for length in first 5 lines of {pub_file}.")
    # ? Look for "length" in the first 5 lines
    row_with_headings = None
    for i in range(len(head)):
        if "length" in head[i].lower():
            row_with_headings = i
            break
    
    logger.info(f"Reading last 5 lines of {pub_file}.")
    # ? Read the last 5 lines of the file
    with open(pub_file, "r") as f:
        tail = f.readlines()[-5:]

    try:
        logger.info(f"Processing {pub_file} into dataframe.")
        # ? Read the CSV file using the row_with_headings and skip last 5 lines
        pub_df = pd.read_csv(
            pub_file,
            skiprows=row_with_headings,
            skipfooter=5,
            engine="python"
        )
    except Exception as e:
        logger.error(f"Could not read pub_0.csv file: {e}")
        return None

    return pub_df

def get_sub_metric_df_from_testdir(test_dir: str, sub_metric: str) -> pd.DataFrame:
    # TODO:
    pass

def get_test_param_df_from_testdir(test_dir: str) -> pd.DataFrame:
    # TODO:
    pass

def get_filepaths_inside_dir(test_dir: str) -> [str]:
    if not os.path.isdir(test_dir):
        logger.error(f"{test_dir} is NOT a directory.")
        return None

    file_paths = []
    for file in os.listdir(test_dir):
        file_paths.append(os.path.join(test_dir, file))

    return file_paths

def get_pub_file_from_testdir(test_dir: str) -> str:
    if not os.path.exists(test_dir):
        logger.error(f"{test_dir} does NOT exist.")
        return None

    test_dir_contents = get_filepaths_inside_dir(test_dir)
    if test_dir_contents == None:
        return None
    
    pub_files = [file for file in test_dir_contents if file.endswith("pub_0.csv")]

    if len(pub_files) == 0:
        logger.error(f"No pub_0.csv files found in {test_dir}.")
        return None

    return pub_files[0]

def main(sys_args: [str] = None) -> None:
    if not sys_args:
        logger.error("No sys args provided.")
        return False
    
    if len(sys_args) != 1:
        logger.error(f"Only one sys arg is allowed. You have provided {len(sys_args)}: {sys_args}")
        return False
    
    if not os.path.exists(sys_args[0]):
        logger.error("File does not exist.")
        return False

    if not os.path.isdir(sys_args[0]):
        logger.error("File is not a directory.")
        return False

    if not os.listdir(sys_args[0]):
        logger.error("Directory is empty.")
        return False
    
    tests_dir_path = sys_args[0]

    # tests_dir_path should be in the format: dir_path/600SEC.../pub0.csv
    # A single test should be in the final folder e.g. 600SEC.../
    # So get the longest path and then go out 2 folders to see all tests.
    # e.g. my_path/some_path/more_folders/600SEC.../pub0.csv
    #   => my_path/some_path/more_folders/

    logger.info(f"Getting longest path for {tests_dir_path}.")
    longest_path = get_longest_path_in_dir(tests_dir_path)

    logger.info(f"Getting test parent dirpath from {longest_path}.")
    test_parent_dirpath = get_test_parent_dirpath_from_fullpath(longest_path)

    test_dirs = [
        os.path.join(
            test_parent_dirpath, 
            dir
        ) for dir in os.listdir(
            test_parent_dirpath
        )
    ]
    logger.info(f"Found {len(test_dirs)} tests in {test_parent_dirpath}.")

    for test_dir in test_dirs:
        logger.info(
            f"[{test_dirs.index(test_dir) + 1}/{len(test_dirs)}] Processing {test_dir}..."
        )
        test_param_df = get_test_param_df_from_testdir(
            test_dir
        )
        latency_df = get_latency_df_from_testdir(
            test_dir
        )
        throughput_df = get_sub_metric_df_from_testdir(
            test_dir, 
            'throughput'
        )
        sample_rate_df = get_sub_metric_df_from_testdir(
            test_dir, 
            'sample_rate'
        )
        lost_samples_df = get_sub_metric_df_from_testdir(
            test_dir, 
            'lost_samples'
        )
        lost_samples_percentage_df = get_sub_metric_df_from_testdir(
            test_dir, 
            'lost_samples_percentage'
        )
        received_samples_df = get_sub_metric_df_from_testdir(
            test_dir, 
            'received_samples'
        )
        received_samples_percentage_df = get_sub_metric_df_from_testdir(
            test_dir, 
            'received_samples_percentage'
        )

        test_df = pd.concat([
            test_param_df,
            latency_df,
            throughput_df,
            sample_rate_df,
            lost_samples_df,
            lost_samples_percentage_df,
            received_samples_df,
            received_samples_percentage_df,
        ], axis=1)

if __name__ == "__main__":
    if pytest.main(["-q", "./pytests", "--exitfirst"]) == 0:
        main(sys.argv[1:])
    else:
        logger.error("Tests failed.")
        sys.exit(1)
