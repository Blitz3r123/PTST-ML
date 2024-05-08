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
from typing import Union

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
def get_longest_path_in_dir(dir_path: str = "") -> str:
    """
    Get the longest path in a directory.
    """
    if dir_path == "":
        logger.error(f"No path passed to get_longest_path_in_dir().")
        return None

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
    """
    Takes a path and returns the third last folder as path.
    e.g.
        phd/year_one/machine_learning/experiments/qos_capture/600SEC_200B/pub.csv

        will give back:

        phd/year_one/machine_learning/experiments/qos_capture
    """

    if "/" not in longest_path:
        logger.error(f"No / found in {longest_path}.")
        return None

    longest_path_items = longest_path.split("/")
    if len(longest_path_items) <= 2:
        logger.error(f"Can't get test parent dirpath of a path with less than 3 nests.")
        logger.error(f"{longest_path}")
        return None

    return "/".join(longest_path_items[:-2])

def get_file_line_count(file_path):
    with open(file_path, "r") as f:
        num_lines = sum(1 for line in f)

    return num_lines

def get_headings_from_pub_file(pub_file: str = "") -> list[str]:
    """
    Reads a csv file,
    looks for the column heading names in the first 10 lines of the file,
    return the list of headings.
    """
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

def get_latency_df_from_testdir(test_dir: str = "") -> pd.DataFrame:
    pub_file = get_pub_file_from_testdir(test_dir)
    if pub_file is None:
        return None

    headings = get_headings_from_pub_file(pub_file)
    if len(headings) == 0:
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
            line_items = [
                item for item in line_items if item != ""
            ]

            if len(line_items) == len(headings):
                number_pattern_regex = re.compile(
                    r'^\d+(\.\d+)?$'
                )

                line_items = [
                    item.strip() for item in line_items
                ]

                items_are_numbers = all([
                    number_pattern_regex.match(
                        item
                    ) for item in line_items
                ])

                if(items_are_numbers):
                    values = [
                        float(item) for item in line_items
                    ]
                    data.append(values)
            else:
                if len(line_items) > 1 and len(headings) > 1:
                    logger.warning(
                        f"Mismatch between column count and line item count."
                    )
                    logger.warning(
                        f"columns: {headings}"
                    )
                    logger.warning(
                        f"Line items: {line_items}"
                    )

    if len(data) == 0:
        logger.error(
            f"No data found when parsing {pub_file}."
        )
        return None
    
    df = pd.DataFrame(data, columns=headings)

    # Only get the latency column and nothing else
    latency_cols = [col for col in df.columns if 'latency' in col.lower()]
    if len(latency_cols) == 0:
        logger.error(
            f"No latency col found for {test_dir}."
        )
        return None

    latency_col = latency_cols[0]
    df = df[latency_col]

    return df

def get_headings_from_sub_file(sub_file: str = "") -> list[str]:
    with open(sub_file, 'r') as f:
        file_line_count = sum(1 for _ in f)

    if file_line_count < 10:
        logger.error(f"Less than 10 lines found in {sub_file}.")
        return []

    with open(sub_file, 'r') as f:
        file_head = [next(f) for x in range(10)]

    for line in file_head:
        if 'length' in line.lower() and 'samples' in line.lower():
            headings = line.strip().split(",")
            headings = [heading.strip() for heading in headings]

    headings = [heading for heading in headings if heading != ""]

    return headings

def get_sub_files_from_testdir(test_dir: str = "") -> pd.DataFrame:
    test_files = os.listdir(test_dir)
    test_files = [file for file in test_files if file.endswith('.csv')]
    test_files = [file for file in test_files if file.startswith('sub')]
    test_files = [os.path.join(test_dir, file) for file in test_files]

    return test_files

def get_sub_metric_df_from_testdir(test_dir: str = "", sub_metric: str = "") -> pd.DataFrame:
    """
    sub_metric could be:
    - total samples
    - samples/s
    - avg samples/s
    - mbps
    - avg mbps
    - lost samples
    - lost samples (%)
    """
    sub_files = get_sub_files_from_testdir(test_dir)

    full_df = pd.DataFrame()

    for sub_file in sub_files:
        sub_name = os.path.basename(sub_file).replace(".csv", "")

        headings = get_headings_from_sub_file(sub_file)
        headings = [_.lower() for _ in headings]
        if sub_metric not in headings:
            logger.error(f"{sub_metric} not found in {sub_file}.")
            continue

        # Add the sub name to the metric names
        # e.g. "total samples" becomes "sub_1 total samples"
        headings = [f"{sub_name} {heading}" for heading in headings]
        headings_count = len(headings)

        with open(sub_file, 'r') as f:
            data = []

            for line in f:
                if 'summary' in line.strip().lower():
                    break

                line_items = line.strip().split(",")
                if len(line_items) == headings_count:
                    number_pattern_regex = re.compile(r'^\d+(\.\d+)?$')
                    line_items = [item.strip() for item in line_items]
                    items_are_numbers = all(
                        [number_pattern_regex.match(item) for item in line_items]
                    )

                    if(items_are_numbers):
                        values = [float(item) for item in line_items]
                        data.append(values)

        sub_df = pd.DataFrame(data, columns=headings)

        if sub_metric == 'total samples':
            sub_df = sub_df[f"{sub_name} total samples"]
        elif sub_metric == 'samples/s':
            sub_df = sub_df[f"{sub_name} samples/s"]
        elif sub_metric == 'avg samples/s':
            sub_df = sub_df[f"{sub_name} avg samples/s"]
        elif sub_metric == 'lost samples':
            sub_df = sub_df[f"{sub_name} lost samples"]
        elif sub_metric == 'lost samples (%)':
            sub_df = sub_df[f"{sub_name} lost samples (%)"]

        full_df = pd.concat([
            full_df,
            sub_df
        ], axis=1)

    full_df[f"total {sub_metric}"] = full_df.sum(axis=1)
    full_df[f"avg {sub_metric} per sub"] = full_df.mean(axis=1)

    full_df = full_df[[
        f"total {sub_metric}",
        f"avg {sub_metric} per sub",
    ]]

    return full_df

def get_test_param_df_from_testdir(test_dir: str = "") -> pd.DataFrame:
    """
    Takes a path to the test folder,
    extracts the parameter values,
    puts it into a dataframe that looks like this:

    duration_sec, datalen_bytes, pub_count, sub_count, use_reliable, use_multicast, durability, latency_count
    ...,            ...,            ...,        ...,    ...,            ...,        ...,            ...

    """
    if test_dir == "":
        logger.error(f"No test_dir passed to get_test_param_df_from_testdir.")
        return None
    
    test_name = get_test_name_from_test_dir(test_dir)
    if test_name is None:
        logger.error(
            f"Couldn't get test_name for {test_dir}."
        )
        return None
    test_name_items = test_name.split("_")

    if len(test_name_items) != 8:
        logger.error(f"{len(test_name_items)} items found instead of 8.")
        return None

    test_param_dict = {}
    for item in test_name_items:
        expected_parameters = [
            'sec',
            'b',
            'p',
            's',
            'be',
            'rel',
            'uc',
            'mc',
            'dur',
            'lc'
        ]
        item_string = re.sub(r'\d+', '', item.lower())
        if item_string not in expected_parameters:
            logger.error(f"Unexpected parameter {item_string} found in {test_name}.")
            return None

        non_numeric_parameters = [
            'uc',
            'mc',
            'be',
            'rel'
        ]
        if item.lower() not in non_numeric_parameters:
            item_value = re.sub(r'\D+', '', item)
            if item_value == '':
                logger.error(
                    f"No value found for {item.lower()} in {test_name}."
                )
                return None

            item_value = int(item_value)

        if 'sec' in item.lower():
            test_param_dict['duration_sec'] = item_value

        elif item.lower().endswith('s'):
            # Could accidentally be a subscriber...
            if 'duration_sec' in test_param_dict:
                test_param_dict['sub_count'] = item_value
            else:
                test_param_dict['duration_sec'] = item_value

        elif item.lower().endswith("p"):
            test_param_dict['pub_count'] = item_value

        elif item.lower().endswith("b"):
            test_param_dict['datalen_byte'] = item_value

        elif item.lower().endswith("dur"):
            test_param_dict['durability'] = item_value

        elif item.lower().endswith("lc"):
            test_param_dict['latency_count'] = item_value

        elif 'uc' in item.lower():
            test_param_dict['use_multicast'] = 0
        elif 'mc' in item.lower():
            test_param_dict['use_multicast'] = 1

        elif 'be' in item.lower():
            test_param_dict['use_reliable'] = 0
        elif 'rel' in item.lower():
            test_param_dict['use_reliable'] = 1

    return pd.DataFrame([test_param_dict])

def get_filepaths_inside_dir(test_dir: str = "") -> [str]:
    if not os.path.isdir(test_dir):
        logger.error(f"{test_dir} is NOT a directory.")
        return None

    file_paths = []
    for file in os.listdir(test_dir):
        file_paths.append(os.path.join(test_dir, file))

    return file_paths

def get_pub_file_from_testdir(test_dir: str = "") -> str:
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

def get_test_name_from_test_dir(test_dir: str = "") -> str:
    """
    Returns the name of the last folder in the path.
    e.g.
        phd/tests/500SEC_433B/
        gives back
        500SEC_433B
    """
    if test_dir is None:
        return None

    if test_dir == "":
        return None

    if test_dir[-1] == "/":
        test_dir = test_dir[:-1]

    test_dir_items = test_dir.split("/")
    test_name = test_dir_items[-1]

    return test_name

def get_distribution_stats_from_col(df: pd.DataFrame) -> dict[str, float]:
    """
    For a given dataframe,
    calculate various statitistical values,
    used to recreate a distribution.

    Statistical values include:
    mean, std, min, max,
    1%, 2%, 5%, 10%,
    20%, 25%,
    30%,
    40%,
    50%,
    60%,
    70%, 75%,
    80%,
    90%, 95%, 98%, 99%
    """
    if not isinstance(df, pd.Series):
        logger.error(f"Dataframe with multiple columns passed instead of series.")
        return None

    distribution_stats = {}

    distribution_stats['mean'] = df.mean()
    distribution_stats['std'] = df.std()
    distribution_stats['min'] = df.min()
    distribution_stats['max'] = df.max()
    distribution_stats['1%'] = df.quantile(0.01)
    distribution_stats['2%'] = df.quantile(0.02)
    distribution_stats['5%'] = df.quantile(0.05)
    distribution_stats['10%'] = df.quantile(0.1)
    distribution_stats['20%'] = df.quantile(0.2)
    distribution_stats['25%'] = df.quantile(0.25)
    distribution_stats['30%'] = df.quantile(0.3)
    distribution_stats['40%'] = df.quantile(0.4)
    distribution_stats['50%'] = df.quantile(0.5)
    distribution_stats['60%'] = df.quantile(0.6)
    distribution_stats['70%'] = df.quantile(0.7)
    distribution_stats['75%'] = df.quantile(0.75)
    distribution_stats['80%'] = df.quantile(0.8)
    distribution_stats['90%'] = df.quantile(0.9)
    distribution_stats['95%'] = df.quantile(0.95)
    distribution_stats['98%'] = df.quantile(0.98)
    distribution_stats['99%'] = df.quantile(0.99)

    return distribution_stats

def get_distribution_stats_df(df: Union[pd.DataFrame, pd.Series], is_latency: bool = False) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        col_names = df.columns
    else:
        col_names = df.name

    if is_latency:
        # Force from a single col df to a series.
        distribution_stats = get_distribution_stats_from_col(df)

        distribution_stats_with_name = {}
        for stat, value in distribution_stats.items():
            distribution_stats_with_name[f"latency (µs) {stat}"] = value

        distribution_stats_df = pd.DataFrame([
            distribution_stats_with_name
        ])

        return distribution_stats_df

    total_col_name = [col for col in col_names if 'total' in col.lower()]
    if len(total_col_name) == 0:
        logger.error(f"No total column found in {col_names}.")
        return None

    avg_col_name = [col for col in col_names if 'per sub' in col.lower()]
    if len(avg_col_name) == 0:
        logger.error(f"No avg per sub column found in {col_names}.")
        return None

    total_col_name = total_col_name[0]
    avg_col_name = avg_col_name[0]

    total_df = df[total_col_name]
    avg_df = df[avg_col_name]

    total_distribution_stats = get_distribution_stats_from_col(total_df)
    total_distribution_stats_with_name = {}
    for stat, value in total_distribution_stats.items():
        total_distribution_stats_with_name[f"{total_col_name} {stat}"] = value

    avg_distribution_stats = get_distribution_stats_from_col(avg_df)
    avg_distribution_stats_with_name = {}
    for stat, value in avg_distribution_stats.items():
        avg_distribution_stats_with_name[f"{avg_col_name} {stat}"] = value

    total_dist_df = pd.DataFrame([total_distribution_stats_with_name])
    avg_dist_df = pd.DataFrame([avg_distribution_stats_with_name])

    df_to_return = pd.concat([
        total_dist_df,
        avg_dist_df
    ], axis=1)

    return df_to_return

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

    logger.info(
        f"Getting longest path for {tests_dir_path}."
    )
    longest_path = get_longest_path_in_dir(
        tests_dir_path
    )
    if longest_path is None:
        logger.error(
            f"Couldn't get longest path for {tests_dir_path}."
        )
        return

    logger.info(
        f"Getting test parent dirpath from {longest_path}."
    )
    test_parent_dirpath = get_test_parent_dirpath_from_fullpath(
        longest_path
    )
    if test_parent_dirpath is None:
        logger.error(
            f"Couldn't get test parent dirpath of {longest_path}."
        )
        return

    test_dirs = [
        os.path.join(
            test_parent_dirpath, 
            dir
        ) for dir in os.listdir(
            test_parent_dirpath
        )
    ]
    logger.info(
        f"Found {len(test_dirs)} tests in {test_parent_dirpath}."
    )
    
    tests_without_results_count = 0

    final_df = pd.DataFrame()

    # TODO: Remove the limiter [:10]
    for test_dir in test_dirs[:10]:
        if not os.path.isdir(test_dir):
            logger.warning(
                f"{test_dir} is NOT a dir. Skipping..."
            )
            continue

        logger.info(
            f"[{test_dirs.index(test_dir) + 1}/{len(test_dirs)}] Processing {test_dir}..."
        )
        param_df = get_test_param_df_from_testdir(test_dir)
        if param_df is None:
            logger.error(
                f"Couldn't get parameters for {test_dir}."
            )
            continue

        latency_df = get_latency_df_from_testdir(
            test_dir
        )
        if latency_df is None:
            logger.error(
                f"No latency results found for {test_dir}."
            )
            tests_without_results_count += 1
            continue
        latency_df = get_distribution_stats_df(latency_df, True)

        throughput_df = get_sub_metric_df_from_testdir(
            test_dir, 
            'mbps'
        )
        throughput_df = get_distribution_stats_df(throughput_df)

        sample_rate_df = get_sub_metric_df_from_testdir(
            test_dir, 
            'samples/s'
        )
        sample_rate_df = get_distribution_stats_df(sample_rate_df)

        lost_samples_df = get_sub_metric_df_from_testdir(
            test_dir, 
            'lost samples'
        )
        lost_samples_df = get_distribution_stats_df(lost_samples_df)

        lost_samples_percentage_df = get_sub_metric_df_from_testdir(
            test_dir, 
            'lost samples (%)'
        )
        lost_samples_percentage_df = get_distribution_stats_df(lost_samples_percentage_df)

        received_samples_df = get_sub_metric_df_from_testdir(
            test_dir, 
            'total samples'
        )
        received_samples_df = get_distribution_stats_df(received_samples_df)

        test_df = pd.concat([
            param_df,
            latency_df,
            throughput_df,
            sample_rate_df,
            lost_samples_df,
            lost_samples_percentage_df,
            received_samples_df,
        ], axis=1)

        final_df = pd.concat([
            test_df, 
            final_df
        ])
        #
        # os.makedirs("./summaries", exist_ok=True)
        # test_name = get_test_name_from_test_dir(test_dir)
        # test_csv_filename = f"./summaries/{test_name}.csv"
        # test_df.to_csv(test_csv_filename, index=False)
        # logger.info(f"Summary written to {test_csv_filename}.")


    final_df_filename = os.path.basename(tests_dir_path)
    final_df.to_csv(f"./{final_df_filename}.csv")
    logger.info(f"Dataset created as {final_df_filename}.csv.")
    logger.info(f"Processed {len(test_dirs)} tests.")
    logger.info(f"{tests_without_results_count} tests failed.")

if __name__ == "__main__":
    if pytest.main(["-q", "./pytests", "--exitfirst"]) == 0:
        main(sys.argv[1:])
    else:
        logger.error("Tests failed.")
        sys.exit(1)
