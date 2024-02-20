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

STATS = [
    'min',
    'max',
    'mean',
    'std',
    '1',
    '2',
    '5',
    '10',
    '25',
    '30',
    '40',
    '50',
    '60',
    '70',
    '75',
    '80',
    '90',
    '95',
    '99',
]

METRICS = [
    'latency', 'throughput', 'sample_rate', 
    'lost_samples', 'lost_samples_percentage',
    'received_samples', 'received_samples_percentage',
]

def calculate_total_and_avg_df(df, column):
    if df is None:
        logger.error("No dataframe provided")
        return None
    
    if column is None:
        logger.error("No column provided")
        return None
    
    similar_columns = [c for c in df.columns if column in c]

    if len(similar_columns) == 0:
        logger.error(f"No similar columns found for column:\n{column}")
        return None

    total_column = f"total_{column}"
    avg_column = f"avg_{column}"

    df[total_column] = df[similar_columns].sum(axis=1)
    df[avg_column] = df[similar_columns].mean(axis=1)

    return df

def get_stats_from_list(values):
    if values is None:
        logger.error("No values provided")
        return None
    
    if len(values) == 0:
        logger.error("Empty values provided")
        return None
    
    stats = {}

    for stat in STATS:
        if stat == "min":
            stats[stat] = np.min(values)
        elif stat == "max":
            stats[stat] = np.max(values)
        elif stat == "mean":
            stats[stat] = np.mean(values)
        elif stat == "std":
            stats[stat] = np.std(values)
        else:
            stats[stat] = values.quantile(int(stat) / 100)

    # ? Look for nans
    for stat in STATS:
        if np.isnan(stats[stat]):
            logger.error(f"NaN found for stat:\n{stat}")
            return None

    return stats

def get_stats(df, metric, cut_off_percentage=0):
    if df is None:
        logger.error("No dataframe provided")
        return None
    
    if metric is None:
        logger.error("No metric provided")
        return None
    
    if metric not in METRICS:
        logger.error(f"Unexpected metric provided:\n{metric}")
        return None
    
    if len(df) == 0:
        logger.error("Empty dataframe provided")
        return None
    
    # ? Remove the first cut_off_percentage of values
    if cut_off_percentage > 0:
        cut_off_count = int(len(df) * (cut_off_percentage / 100))
        df = df.iloc[cut_off_count:]

    metric_stats = {}
    if metric == "latency":
        metric_stats = get_stats_from_list(df['latency_us'])
        if metric_stats is None:
            return None
        
        # ? Prefix the latency stats with latency_{stat}
        for stat in STATS:
            metric_stats[f"latency_us_{stat}"] = metric_stats.pop(stat)
        
        return metric_stats
    
    elif metric == "throughput":
        column = "throughput_mbps"
    
    elif metric == "sample_rate":
        column = "samples_per_sec"

    elif metric == "lost_samples":
        column = "lost_samples"

    elif metric == "lost_samples_percentage":
        column = "lost_samples_percentage"

    elif metric == "received_samples":
        column = "received_samples"

    elif metric == "received_samples_percentage":
        column = "received_samples_percentage"

    else:
        logger.error(f"Unexpected metric provided:\n{metric}")
        return None

    total_df = calculate_total_and_avg_df(df, column)
    if total_df is None:
        return None
    
    total_metric_stats = get_stats_from_list(total_df[f"total_{column}"])
    if total_metric_stats is None:
        return None
    
    # ? Prefix the total stats with total_{metric}_{stat}
    for stat in STATS:
        total_metric_stats[f"total_{column}_{stat}"] = total_metric_stats.pop(stat)

    avg_metric_stats = get_stats_from_list(total_df[f"avg_{column}"])
    if avg_metric_stats is None:
        return None
    
    # ? Prefix the avg stats with avg_{metric}_{stat}
    for stat in STATS:
        avg_metric_stats[f"avg_{column}_{stat}"] = avg_metric_stats.pop(stat)

    metric_stats = {**total_metric_stats, **avg_metric_stats}

    return metric_stats

def get_input_variables_from_testpath(testpath):
    if testpath is None:
        logger.error("No testpath provided")
        return None

    if not os.path.isfile(testpath):
        logger.error(f"Testpath does not exist:\n{testpath}")
        return None
    
    testpath = os.path.basename(testpath)
    testpath = testpath.replace(".csv", "")

    expected_variables = [
        "duration_sec",
        'datalen_bytes',
        'pub_count',
        'sub_count',
        'reliability',
        'multicast',
        'durability',
        'latency_count'
    ]

    testpath_elements = testpath.split("_")

    if len(testpath_elements) != len(expected_variables):
        logger.error(f"Unexpected number of variables in testpath:\n{testpath}\nExpected: {expected_variables}")
        return None
    
    input_variables = {}
    for i, variable in enumerate(expected_variables):
        element_value = testpath_elements[i]

        if variable == "duration_sec":
            if "sec" not in element_value.lower():
                logger.error(f"Unexpected value for duration_sec:\n{element_value}")
                return None
            
            duration_sec = element_value.lower().replace("sec", "")
            
            if not duration_sec.isnumeric():
                logger.error(f"Unexpected value for duration_sec:\n{element_value}")
                return None
            
            input_variables[variable] = int(duration_sec)

        elif variable == "datalen_bytes":
            
            if "b" not in element_value.lower():
                logger.error(f"Unexpected value for datalen_bytes:\n{element_value}")
                return None
            
            datalen_bytes = element_value.lower().replace("b", "")

            if not datalen_bytes.isnumeric():
                logger.error(f"Unexpected value for datalen_bytes:\n{element_value}")
                return None
            
            input_variables[variable] = int(datalen_bytes)

        elif variable == "pub_count":

            if "p" not in element_value.lower():
                logger.error(f"Unexpected value for pub_count:\n{element_value}")
                return None
            
            pub_count = element_value.lower().replace("p", "")

            if not pub_count.isnumeric():
                logger.error(f"Unexpected value for pub_count:\n{element_value}")
                return None
            
            input_variables[variable] = int(pub_count)

        elif variable == "sub_count":

            if "s" not in element_value.lower():
                logger.error(f"Unexpected value for sub_count:\n{element_value}")
                return None
            
            sub_count = element_value.lower().replace("s", "")

            if not sub_count.isnumeric():
                logger.error(f"Unexpected value for sub_count:\n{element_value}")
                return None
            
            input_variables[variable] = int(sub_count)

        elif variable == "reliability":

            valid_values = ['be', 'rel']

            if element_value.lower() not in valid_values:
                logger.error(f"Unexpected value for reliability:\n{element_value}")
                return None
            
            input_variables[variable] = element_value.lower() == "rel"

        elif variable == "multicast":

            valid_values = ['uc', 'mc']

            if element_value.lower() not in valid_values:
                logger.error(f"Unexpected value for multicast:\n{element_value}")
                return None
            
            input_variables[variable] = element_value.lower() == "mc"

        elif variable == "durability":

            if "dur" not in element_value.lower():
                logger.error(f"Unexpected value for durability:\n{element_value}")
                return None
            
            durability = element_value.lower().replace("dur", "")

            if not durability.isnumeric():
                logger.error(f"Unexpected value for durability:\n{element_value}")
                return None
            
            input_variables[variable] = int(durability)

        elif variable == "latency_count":

            if "lc" not in element_value.lower():
                logger.error(f"Unexpected value for latency_count:\n{element_value}")
                return None
            
            latency_count = element_value.lower().replace("lc", "")

            if not latency_count.isnumeric():
                logger.error(f"Unexpected value for latency_count:\n{element_value}")
                return None
            
            input_variables[variable] = int(latency_count)

        else:
            logger.error(f"Unexpected variable:\n{variable}")
            return None
        
    return input_variables

def main():
    logger.remove(0)
    logger.add(sys.stderr, format="{time} {message}", level="ERROR")
    logger.add("errors.log", format="{time} {message}", level="ERROR")
    logger.add("warnings.log", format="{time} {message}", level="WARNING")
    
    if len(sys.argv) < 3:
        logger.error("No raw data directory's directory provided")
        return

    summaries_parent_dir = sys.argv[1]

    if not os.path.isdir(summaries_parent_dir):
        logger.error(f"Summaries parent directory does not exist:\n{summaries_parent_dir}")
        return
    
    summaries_dirs = [os.path.join(summaries_parent_dir, f) for f in os.listdir(summaries_parent_dir) if os.path.isdir(os.path.join(summaries_parent_dir, f))]

    if len(summaries_dirs) == 0:
        logger.error(f"No summaries directories found in directory:\n{summaries_parent_dir}")
        return
    
    console.print(f"Found {len(summaries_dirs)} directories to summarise.", style="bold green")
    for summaries_dir in summaries_dirs:

        summaries_dir_index = summaries_dirs.index(summaries_dir) + 1

        if not os.path.isdir(summaries_dir):
            logger.error(f"Summaries directory does not exist:\n{summaries_dir}")
            return
        
        cut_off_percentage = sys.argv[2] if len(sys.argv) > 2 else 0
        cut_off_percentage = float(cut_off_percentage)

        summaries_dirname = os.path.basename(summaries_dir)

        summaries = [os.path.join(summaries_dir, f) for f in os.listdir(summaries_dir) if f.endswith(".csv")]

        if len(summaries) == 0:
            logger.error(f"No summaries found in directory:\n{summaries_dir}")
            return
        
        dataset_df = pd.DataFrame()

        for summary in track(
            summaries, 
            description=f"[{summaries_dir_index}/{len(summaries_dirs)}] Summarising {os.path.basename(summaries_dir)}..."
        ):
            input_variables = get_input_variables_from_testpath(summary)
            
            if input_variables is None:
                continue

            summary_df = pd.read_csv(summary)

            if len(summary_df) == 0:
                logger.error(f"Empty summary:\n{summary}")
                continue

            all_metric_stats = {}
            for METRIC in METRICS:
                metric_stats = get_stats(summary_df, METRIC, cut_off_percentage)

                if metric_stats is None:
                    continue

                all_metric_stats = {**all_metric_stats, **metric_stats}

            all_metric_stats = {**input_variables, **all_metric_stats}

            new_row = pd.DataFrame(all_metric_stats, index=[0])

            dataset_df = pd.concat([dataset_df, new_row], ignore_index=True)
        
        dataset_df['reliability'] = dataset_df['reliability'].astype(int)
        dataset_df['multicast'] = dataset_df['multicast'].astype(int)

        # ? One hot encode durability column
        dataset_df = pd.get_dummies(dataset_df, columns=['durability'], prefix=['durability'])

        durability_columns = [c for c in dataset_df.columns if "durability" in c]
        for durability_column in durability_columns:
            dataset_df[durability_column] = dataset_df[durability_column].astype(int)

        # ? Reorder to put input variables first
        inputs = [
            'duration_sec',
            'datalen_bytes',
            'pub_count',
            'sub_count',
            'reliability',
            'multicast',
            'latency_count',
            'durability_0',
            'durability_1',
            'durability_2',
            'durability_3',
        ]
        outputs = [c for c in dataset_df.columns if c not in inputs]
        dataset_df = dataset_df[inputs + outputs]

        truncate_suffix = "_truncated" if cut_off_percentage > 0 else ""

        dataset_df.to_csv(f"{summaries_dirname}{truncate_suffix}_dataset.csv", index=False)

        console.print(f"Dataset created:\n\t{summaries_dirname}{truncate_suffix}_dataset.csv", style="bold green")

main()