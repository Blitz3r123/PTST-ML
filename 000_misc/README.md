# Available Scripts

1. `summaries_from_raw_data.py`
2. `summaries_from_raw_data_dir.py`
3. `dataset_from_summaries.py`
4. `dataset_from_summaries_dir.py`
5. `dataset_combiner.py`

---

# 1. `summaries_from_raw_data.py`

Takes a folder of test folders and turns into a folder of test files. Summarises all files from a test into a single file.

## Usage:
```sh
py summaries_from_raw_data.py <path_to_dir_containing_tests>
```

Here is what `path_to_dir_containing_tests` should look like:

- 📂 `2023-09-18_qos_capture_rcg_ps_raw`
    - 📂 `600SEC_127915B_1P_22S_BE_UC_2DUR_100LC`
      - 📄 `pub_0.csv`
      - 📄 `sub_0.csv`
      - ...
      - 📄 `sub_10.csv`
    - ...
    - 📂 `600SEC_127915B_1P_22S_BE_UC_2DUR_100LC`

---

# 2. `summaries_from_raw_data_dir.py`

Takes a folder of folders of test folders and turns into a folder of folders of test files. Summarises all files from a test into a single file.

## Usage:
```sh
py summaries_from_raw_data.py <path_to_dir_containing_dir_containing_tests>
```

---
# 3. `dataset_from_summaries.py`

Creates a single summary file containing the distribution recreation stats of each test. Summarises a folder of test files into a single file. Has the option to cut out a starting percentage of data to remove warm up measurements.

## Usage:
```sh
py dataset_from_summaries.py <path_to_summaries> <starting_cut_off_percentage>
```

---

# 4. `dataset_from_summaries_dir.py`

Creates multiple summary files containing the distribution recreation stats of each test for each campaign. Summarises a folder folders of test files into a folder of files. Has the option to cut out a starting percentage of data to remove warm up measurements.

## Usage:
```sh
py dataset_from_summaries.py <path_to_path_to_summaries> <starting_cut_off_percentage>
```

---

# 5. `dataset_combiner.py`

Combines multiple csv files into one. Takes in a folder of csv files and generates a single csv file.

## Usage:
```sh
py dataset_combiner.py <path_to_datasets>
```