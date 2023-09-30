import sys
import os
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.progress import track
from icecream import ic
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

console = Console()

def show_dataset_stats(csv_files):
    if csv_files is None:
        logger.error(f"No csv files sent.")
        return None
    
    dataset_info = []
    total_row_count = 0
    for csv_file in csv_files:
        dataset_name = os.path.basename(csv_file)
        df = pd.read_csv(csv_file)

        row_count = len(df)
        column_count = len(df.columns)

        total_row_count += row_count

        dataset_info.append({
            "dataset_name": dataset_name,
            "row_count": row_count,
            "column_count": column_count
        })

    dataset_info_df = pd.DataFrame(dataset_info)
    
    if len(dataset_info_df) == 0:
        logger.error(f"No datasets found.")
        return None
    
    stat_table = Table(show_header=True, header_style="bold magenta")
    stat_table.add_column("Dataset Name")
    stat_table.add_column("Row Count")
    stat_table.add_column("Column Count")

    for index, row in dataset_info_df.iterrows():
        
        if index == len(dataset_info_df) - 1:
            end_section = True
        else:
            end_section = False

        stat_table.add_row(
            str(row["dataset_name"]), 
            str(row["row_count"]), 
            str(row["column_count"]),
            end_section=end_section
        )

    stat_table.add_row("combined_dataset.csv", str(total_row_count), str( dataset_info_df["column_count"].max() ))

    console.print(stat_table)

def main():
    if len(sys.argv) < 2:
        logger.error("Please specify the folder of the datasets.")
        return

    dataset_dir = sys.argv[1]

    if not os.path.isdir(dataset_dir):
        logger.error("The dataset folder does not exist.")
        return
    
    csv_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(".csv")]

    if len(csv_files) == 0:
        logger.error("The dataset folder does not contain any csv files.")
        return
    
    show_dataset_stats(csv_files)

    new_df = pd.DataFrame()
    for csv_file in track(csv_files, description="Combining datasets..."):
        df = pd.read_csv(os.path.join(dataset_dir, csv_file))
        new_df = pd.concat([new_df, df], ignore_index=True)

    # ? Remove duplicates
    before_removing_duplicates = len(new_df)
    new_df = new_df.drop_duplicates()
    after_removing_duplicates = len(new_df)

    console.print(f"Removed [bold red]{before_removing_duplicates - after_removing_duplicates}[/bold red] duplicate rows - new length: {after_removing_duplicates}.")

    new_df.to_csv("combined_dataset.csv", index=False)

    console.print(f"Combined dataset saved to [bold green]combined_dataset.csv[/bold green]")

main()