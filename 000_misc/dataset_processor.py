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
import re
import pandas as pd
from icecream import ic
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    filename="dataset_processor.log", 
    filemode="w",
    format='%(asctime)s \t%(levelname)s \t%(message)s'
)
logger = logging.getLogger(__name__)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s \t%(levelname)s \t%(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

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
                if "leftovers" not in file_path.lower() and "logs" not in file_path.lower():
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

def get_headings_from_pub_file(pub_file: str) -> list[str]:
    if pub_file is None:
        return []

    if not os.path.exists(pub_file):
        logger.error(f"{pub_file} is not a valid path.")
        return []

    with open(pub_file, 'r') as f:
        file_line_count = sum(1 for _ in f)

    if file_line_count < 10:
        logger.error(f"Less than 10 lines found in {pub_file}.")
        return []

    with open(pub_file, 'r') as f:
        file_head = [next(f) for x in range(10)]

    for line in file_head:
        if 'length' in line.lower() and 'latency' in line.lower():
            headings = line.strip().split(",")
            headings = [heading.strip() for heading in headings]

    headings = [heading for heading in headings if heading != ""]

    return headings

def get_latency_df_from_testdir(test_dir: str) -> pd.DataFrame:
    logger.info(f"Getting latency df from {test_dir}.")
    pub_file = get_pub_file_from_testdir(test_dir)
    if pub_file is None:
        return None

    headings = get_headings_from_pub_file(pub_file)

    headings_count = len(headings)

    if headings_count == 0:
        logger.error(f"No headings found in pub_0.csv of {test_dir}.")
        return None

    with open(pub_file, 'r') as f:
        data = []
        for line in f:
            # Here we've reached the end of the results
            # so we don't need to read the rest of the file
            if 'summary' in line.strip().lower():
                break

            line_items = line.strip().split(",")

            if len(line_items) == headings_count:
                number_pattern_regex = re.compile(r'^\d+(\.\d+)?$')
                line_items = [item.strip() for item in line_items]
                items_are_numbers = all([number_pattern_regex.match(item) for item in line_items])

                if(items_are_numbers):
                    values = [float(item) for item in line_items]
                    data.append(values)

    df = pd.DataFrame(data, columns=headings)
    df.drop(
        columns=[
            'Length (Bytes)',
            'Ave (μs)',
            'Std (μs)',
            'Min (μs)',
            'Max (μs)'
        ],
        inplace=True
    )
    df.reset_index(
        drop=True, 
        inplace=True
    )
    df.rename(columns={"Latency (μs)": "latency_us"})

    return df

def get_sub_metric_df_from_testdir(test_dir: str, sub_metric: str) -> pd.DataFrame:
    # TODO:
    pass

def get_test_param_df_from_testdir(test_dir: str) -> pd.DataFrame:
    test_name = get_test_name_from_test_dir(test_dir)
    test_name_items = test_name.split("_")

    if 'sec' in test_name_items[0].lower():
        duration_sec = test_name_items[0].lower().replace("sec", "")
    else:
        duration_sec = test_name_items[0].lower().replace("s", "")

    try:
        duration_sec = int(duration_sec)
    except Exception as e:
        logger.error(f"{e}: {test_name}")

    datalen_byte = int(test_name_items[1].lower().replace("b", ""))
    pub_count = int(test_name_items[2].lower().replace("p", ""))
    sub_count = int(test_name_items[3].lower().replace("s", ""))
    durability = int(test_name_items[6].lower().replace("dur", ""))
    latency_count = int(test_name_items[7].lower().replace("lc", ""))

    if test_name_items[4].lower() == 'be':
        use_reliable = 0
    elif test_name_items[4].lower() == 'rel':
        use_reliable = 1
    else:
        logger.warning(f"Unknown item found when parsing reliability usage for {test_name}:\t{test_name_items[4]}")

    if test_name_items[5].lower() == 'uc':
        use_multicast = 0
    elif test_name_items[5].lower() == 'mc':
        use_multicast = 1
    else:
        logger.warning(f"Unknown item found when parsing multicast usage for {test_name}:\t{test_name_items[5]}")

    df = pd.DataFrame(
        [[
            duration_sec,
            datalen_byte,
            pub_count,
            sub_count,
            use_reliable,
            use_multicast,
            durability,
            latency_count
        ]],
        columns=[
            'duration_sec',
            'datalen_byte',
            'pub_count',
            'sub_count',
            'use_reliable',
            'use_multicast',
            'durability',
            'latency_count'

        ]
    )

    return df

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

def get_test_name_from_test_dir(test_dir: str) -> str:
    if test_dir[-1] == "/":
        test_dir = test_dir[:-1]

    test_dir_items = test_dir.split("/")
    test_name = test_dir_items[-1]

    return test_name

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
    
    tests_without_results_count = 0

    for test_dir in test_dirs:
        if not os.path.isdir(test_dir):
            continue

        logger.info(
            f"[{test_dirs.index(test_dir) + 1}/{len(test_dirs)}] Processing {test_dir}..."
        )

        latency_df = get_latency_df_from_testdir(
            test_dir
        )
        if latency_df is None:
            logger.error(f"No latency results found for {test_dir}.")
            tests_without_results_count += 1
            continue

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
            latency_df,
            throughput_df,
            sample_rate_df,
            lost_samples_df,
            lost_samples_percentage_df,
            received_samples_df,
            received_samples_percentage_df,
        ], axis=1)

        os.makedirs("./summaries", exist_ok=True)

        test_name = get_test_name_from_test_dir(test_dir)
        test_csv_filename = f"./summaries/{test_name}.csv"
        test_df.to_csv(test_csv_filename, index=False)

if __name__ == "__main__":
    if pytest.main(["-q", "./pytests", "--exitfirst"]) == 0:
        main(sys.argv[1:])
    else:
        logger.error("Tests failed.")
        sys.exit(1)
