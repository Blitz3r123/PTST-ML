import sys
import os
from loguru import logger
from rich.console import Console
from rich.progress import track
from icecream import ic
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

console = Console()

def get_expected_duration_sec_from_testname(testname):
    if testname is None:
        logger.error("No test name provided")
        return None
    
    testname = os.path.basename(testname)

    if "sec" not in testname.lower():
        logger.error(f"SEC not found in: {testname}")
        return None
    
    substring = testname.lower().split("sec")[0]

    if not substring.isdigit():
        logger.error(f"Could not get expected duration from: {testname}")
        return None
    
    return int(substring)

def calculate_received_samples_percentage(received_samples_df, lost_samples_df):
    if received_samples_df is None:
        logger.error("No received samples dataframe provided")
        return None
    
    if lost_samples_df is None:
        logger.error("No lost samples dataframe provided")
        return None
    
    if len(received_samples_df.columns) != len(lost_samples_df.columns):
        logger.error("Number of columns in received samples dataframe does not match number of columns in lost samples dataframe")
        return None
    
    cols = list(received_samples_df.columns)
    cols = [col.replace("_received_samples", "") for col in cols]
    unique_cols = np.unique(cols)

    if len(unique_cols) == 0:
        logger.error("Could not get unique columns from received samples dataframe")
        return None
    
    received_samples_percentage_df = pd.DataFrame()

    for col in unique_cols:
        received_samples = received_samples_df[f"{col}_received_samples"].dropna().astype(int, errors="ignore")

        lost_samples = lost_samples_df[f"{col}_lost_samples"].dropna().astype(int, errors="ignore")

        total_samples = received_samples + lost_samples

        try:
            received_samples_percentage = received_samples / total_samples * 100
        except Exception as e:
            logger.error(f"Could not calculate received samples percentage for {col}: {e}")

            return None
        
        new_col = pd.Series(received_samples_percentage.values, name=f"{col}_received_samples_percentage")

        received_samples_percentage_df = pd.concat([received_samples_percentage_df, new_col], axis=1)

    if received_samples_percentage_df is None:
        logger.error("Could not get received samples percentage dataframe")
        return None
    
    return received_samples_percentage_df

def get_sub_metric_df(test_dir, metric):
    if test_dir is None or metric is None:
        logger.error("No test directory or metric provided")
        return None
    
    if not os.path.exists(test_dir):
        logger.error(f"Test directory does not exist: {test_dir}")
        return None
    
    csv_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".csv")]

    if len(csv_files) == 0:
        logger.error(f"No CSV files found in {test_dir}")
        return None
    
    sub_files = [f for f in csv_files if "sub" in f]
    sub_csv_files = [f for f in sub_files if f.endswith(".csv")]

    if len(sub_csv_files) == 0:
        logger.error(f"No sub CSV files found in {test_dir}")
        return None
    
    metric_df = pd.DataFrame()

    expected_duration_sec = get_expected_duration_sec_from_testname(test_dir)

    if expected_duration_sec is None:
        logger.error(f"Could not get expected duration in seconds from test name: {test_dir}")
        return None

    for sub_csv_file in sub_csv_files:
        # ? Read the first 5 lines of the file
        with open(sub_csv_file, "r") as f:
            head = [next(f) for x in range(5)]

        if len(head) == 0:
            logger.error(f"sub CSV file is empty: {sub_csv_file}")
            return None

        # ? Look for "length" in the first 5 lines
        row_with_headings = None
        for i in range(len(head)):
            if "length" in head[i].lower():
                row_with_headings = i
                break
        
        if row_with_headings is None:
            logger.error(f"Could not find row with metric headings in sub CSV file: {sub_csv_file}")
            return None
        
        sub_df = pd.read_csv(sub_csv_file, skiprows=row_with_headings, skipfooter=5, engine="python")

        # ? Check that the number of rows is approximately equal to the expected duration in seconds
        if len(sub_df) < expected_duration_sec * 0.9:
            logger.error(f"Number of rows in sub CSV file is less than 90% of expected duration in seconds: {sub_csv_file}")
            return None
        
        if len(sub_df) > expected_duration_sec * 1.1:
            logger.error(f"Number of rows in sub CSV file is greater than 110% of expected duration in seconds: {sub_csv_file}")
            return None

        if sub_df is None:
            logger.error(f"Could not read sub CSV file: {sub_csv_file}")
            return None
        
        sub_cols = sub_df.columns
        sub_name = os.path.basename(sub_csv_file).split(".")[0]
        
        if metric == "throughput":
            cols = [c for c in sub_cols if "mbps" in c.lower() and "avg" not in c.lower()]
            new_name = f"{sub_name}_throughput_mbps"

        elif metric == "sample_rate":
            cols = [c for c in sub_cols if "samples/s" in c.lower()]
            new_name = f"{sub_name}_samples_per_sec"
        
        elif metric == "lost_samples":
            cols = [c for c in sub_cols if "lost samples" in c.lower() and "%" not in c.lower()]
            new_name = f"{sub_name}_lost_samples"
        
        elif metric == "lost_samples_percentage":
            cols = [c for c in sub_cols if "lost samples" in c.lower() and "%" in c.lower()]
            new_name = f"{sub_name}_lost_samples_percentage"

        elif metric == "received_samples":
            cols = [c for c in sub_cols if "total samples" in c.lower()]
            new_name = f"{sub_name}_received_samples"

        else:
            logger.error(f"Invalid metric provided: {metric}")
            return None

        if len(cols) == 0:
            logger.error(f"Could not find {metric} columns in sub_df for {sub_csv_file}")
            return None
        
        metric_col = cols[0]
        sub_df = sub_df[metric_col]
        sub_df.dropna(inplace=True)
        
        if len(sub_df) == 0:
            logger.error(f"Could not get throughput_df for {sub_csv_file}")
            return None

        new_col = pd.Series(sub_df.values, name=new_name)
        metric_df = pd.concat([metric_df, new_col], axis=1)

    if metric_df is None:
        logger.error(f"Could not get metric_df for {test_dir}")
        return None
    
    if len(metric_df.columns) != len(sub_csv_files):
        logger.error(f"Number of columns ({len(metric_df.columns)}) in metric_df does not match number of sub CSV files ({len(sub_csv_files)}) for {test_dir}")
        return None
    
    return metric_df

def get_pub_df(pub_0_csv_file):
    if pub_0_csv_file is None:
        logger.error("No pub_0.csv file provided")
        return None
    
    if not os.path.exists(pub_0_csv_file):
        logger.error(f"pub_0.csv file does not exist: {pub_0_csv_file}")
        return None
    
    # ? Read the first 5 lines of the file
    with open(pub_0_csv_file, "r") as f:
        head = [next(f) for x in range(5)]

    if len(head) == 0:
        logger.error(f"pub_0.csv file is empty: {pub_0_csv_file}")
        return None
    
    # ? Look for "length" in the first 5 lines
    row_with_headings = None
    for i in range(len(head)):
        if "length" in head[i].lower():
            row_with_headings = i
            break

    if row_with_headings is None:
        logger.error(f"Could not find row with metric headings in pub_0.csv file: {pub_0_csv_file}")
        return None
    
    # ? Read the last 5 lines of the file
    with open(pub_0_csv_file, "r") as f:
        tail = f.readlines()[-5:]

    if len(tail) == 0:
        logger.error(f"pub_0.csv file is empty: {pub_0_csv_file}")
        return None
    
    try:
        # ? Read the CSV file using the row_with_headings and skip last 5 lines
        pub_df = pd.read_csv(pub_0_csv_file, skiprows=row_with_headings, skipfooter=5, engine="python")
    except Exception as e:
        logger.error(f"Could not read pub_0.csv file: {e}")
        return None

    if pub_df is None:
        logger.error(f"Could not read pub_0.csv file: {pub_0_csv_file}")
        return None
    
    return pub_df

def get_latency_df(test_dir):
    if test_dir is None:
        logger.error("No test directory provided")
        return None
    
    if not os.path.exists(test_dir):
        logger.error(f"Test directory does not exist: {test_dir}")
        return None
    
    pub_0_csv_file = os.path.join(test_dir, "pub_0.csv")

    if not os.path.exists(pub_0_csv_file):
        logger.error(f"pub_0.csv does not exist in {test_dir}")
        return None
    
    pub_df = get_pub_df(pub_0_csv_file)

    if pub_df is None:
        logger.error(f"Could not get pub_df for {test_dir}")
        return None
    
    pub_cols = pub_df.columns
    latency_cols = [c for c in pub_cols if "latency" in c.lower()]

    if len(latency_cols) == 0:
        logger.error(f"Could not find latency columns in pub_df for {test_dir}")
        return None
    
    latency_col = latency_cols[0]
    latency_df = pub_df[latency_col]
    
    if latency_df is None:
        logger.error(f"Could not get latency_df for {test_dir}")
        return None
    
    latency_df.dropna(inplace=True)
    latency_df.rename("latency_us", inplace=True)

    return latency_df

def get_expected_participant_count(testname, type):
    if testname is None:
        logger.error("No test name provided")
        return None
    
    if type is None:
        logger.error("No type provided")
        return None

    valid_types = ["pub", "sub"]

    if type not in valid_types:
        logger.error(f"Invalid type provided for {testname}: {type}")
        return None
    
    if type == "pub":
        search_str = "P_"
    elif type == "sub":
        search_str = "S_"
    else:
        logger.error(f"Invalid type provided for {testname}: {type}")
        return None
    
    if search_str not in testname:
        logger.error(f"Invalid test name provided: {testname}")
        return None
    
    substring = testname.split(search_str)[0]

    if "_" not in substring:
        logger.error(f"Invalid test name provided: {testname}")
        return None
    
    substring = substring.split("_")[-1]

    if not substring.isdigit():
        logger.error(f"Invalid test name provided: {testname}")
        return None
    
    return int(substring)

def has_expected_files(test_dir):
    if test_dir is None:
        logger.error("No test directory provided")
        return False

    if not os.path.exists(test_dir):
        logger.error(f"Test directory does not exist: {test_dir}")
        return False

    testname = os.path.basename(test_dir)
    expected_sub_count = get_expected_participant_count(testname, "sub")

    if expected_sub_count is None:
        return False
    
    expected_csv_count = expected_sub_count + 1

    csv_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".csv")]

    if len(csv_files) != expected_csv_count:
        return False
        
    empty_csv_files = [f for f in csv_files if os.path.getsize(f) == 0]

    if len(empty_csv_files) > 0:
        logger.warning(f"Test directory {test_dir} has empty CSV files: {empty_csv_files}")
        return False

    # ? Check if pub_0.csv is in csv_files
    pub_csv_files = [f for f in csv_files if "pub" in f]
    pub_0_csv_files = [f for f in pub_csv_files if "pub_0.csv" in f]

    if len(pub_0_csv_files) != 1:
        logger.warning(f"Test directory {test_dir} does not have pub_0.csv")
        return False
    
    sub_files = [f for f in csv_files if "sub" in f]
    sub_csv_files = [f for f in sub_files if f.endswith(".csv")]

    if len(sub_csv_files) != expected_sub_count:
        logger.warning(f"Test directory {test_dir} does not have expected number of sub CSV files")
        return False

    return True

def main():
    logger.remove(0)
    logger.add(sys.stderr, format="{time} {message}", level="CRITICAL")
    logger.add("errors.log", format="{time} {message}", level="ERROR")
    logger.add("warnings.log", format="{time} {message}", level="WARNING")

    if len(sys.argv) != 2:
        logger.error("No raw data directory's directory provided")
        return

    raw_data_dir_dir = sys.argv[1]

    if raw_data_dir_dir is None:
        logger.error("No raw data directory's directory provided")
        return

    if not os.path.exists(raw_data_dir_dir):
        logger.error(f"Raw data directory's directory does not exist: {raw_data_dir_dir}")
        return
    
    raw_data_dirs = [os.path.join(raw_data_dir_dir, f) for f in os.listdir(raw_data_dir_dir) if os.path.isdir(os.path.join(raw_data_dir_dir, f))]

    if len(raw_data_dirs) == 0:
        logger.error(f"No raw data directories found in {raw_data_dir_dir}")
        return
    
    console.print(f"\nFound {len(raw_data_dirs)} raw data directories.\n", style="bold green")

    for raw_data_dir in raw_data_dirs:

        raw_data_dir_index = raw_data_dirs.index(raw_data_dir) + 1

        if not os.path.exists(raw_data_dir):
            logger.error(f"Raw data directory does not exist: {raw_data_dir}")
            return

        test_dirs = [os.path.join(raw_data_dir, f) for f in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, f))]

        if len(test_dirs) == 0:
            logger.error(f"No test directories found in {raw_data_dir}")
            return
        
        for test_dir in track(test_dirs, description=f"[{raw_data_dir_index}/{len(raw_data_dirs)}] Summarising tests in {os.path.basename(raw_data_dir)}..."):
            if not has_expected_files(test_dir):
                logger.warning(f"Test directory {test_dir} does not have expected files")
                continue

            latency_df = get_latency_df(test_dir)

            if latency_df is None:
                logger.warning(f"Test directory {test_dir} does not have latency data")
                continue

            throughput_df = get_sub_metric_df(test_dir, "throughput")

            if throughput_df is None:
                logger.warning(f"Test directory {test_dir} does not have throughput data")
                continue

            sample_rate_df = get_sub_metric_df(test_dir, "sample_rate")

            if sample_rate_df is None:
                logger.warning(f"Test directory {test_dir} does not have sample rate data")
                continue

            lost_samples_df = get_sub_metric_df(test_dir, "lost_samples")

            if lost_samples_df is None:
                logger.warning(f"Test directory {test_dir} does not have lost samples data")
                continue

            lost_samples_percentage_df = get_sub_metric_df(test_dir, "lost_samples_percentage")

            if lost_samples_percentage_df is None:
                logger.warning(f"Test directory {test_dir} does not have lost samples percentage data")
                continue

            received_samples_df = get_sub_metric_df(test_dir, "received_samples")

            if received_samples_df is None:
                logger.warning(f"Test directory {test_dir} does not have received samples data")
                continue

            received_samples_percentage_df = calculate_received_samples_percentage(received_samples_df, lost_samples_df)

            if received_samples_percentage_df is None:
                logger.warning(f"Test directory {test_dir} does not have received samples percentage data")
                continue

            test_df = pd.concat([
                latency_df, 
                throughput_df, 
                sample_rate_df, 
                lost_samples_df, 
                lost_samples_percentage_df, 
                received_samples_df, 
                received_samples_percentage_df
            ], axis=1)

            if test_df is None:
                logger.warning(f"Test directory {test_dir} does not have test data")
                continue

            test_name = os.path.basename(test_dir)
            camp_dirname = os.path.basename( os.path.dirname(test_dir) )
            os.makedirs(f"{camp_dirname}_summaries", exist_ok=True)
            test_df.to_csv(f"{camp_dirname}_summaries/{test_name}.csv", index=False)

        console.print(f"Summaries saved to {os.path.basename(raw_data_dir)}_summaries", style="bold green")

main()