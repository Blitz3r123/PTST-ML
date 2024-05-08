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

def get_headings_from_pub_file(pub_file: str = "") -> list[str]:
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
    if test_dir == "":
        logger.error(f"No test_dir passed to get_test_param_df_from_testdir.")
        return None

    if not os.path.exists(test_dir):
        logger.error(f"{test_dir} does NOT exist.")
        return None

    if not os.path.isdir(test_dir):
        logger.error(f"{test_dir} is NOT a directory.")
        return None

    test_name = get_test_name_from_test_dir(test_dir)
    test_name_items = test_name.split("_")

    if len(test_name_items) != 8:
        logger.error(f"{len(test_name_items)} items found instead of 8.")
        return None

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

def get_distribution_stats_df(df: pd.DataFrame, is_latency: bool = False) -> pd.DataFrame:
    col_names = df.columns

    if is_latency:
        # Force from a single col df to a series.
        df = df.iloc[:, 0]
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

    final_df = pd.DataFrame()

    for test_dir in test_dirs[:10]:
        if not os.path.isdir(test_dir):
            continue

        logger.info(
            f"[{test_dirs.index(test_dir) + 1}/{len(test_dirs)}] Processing {test_dir}..."
        )
        param_df = get_test_param_df_from_testdir(test_dir)

        latency_df = get_latency_df_from_testdir(
            test_dir
        )
        if latency_df is None:
            logger.error(f"No latency results found for {test_dir}.")
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
