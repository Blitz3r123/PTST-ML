# Dataset Processor

A script that takes in a bunch of tests and turns it into a single spreadsheet.

# Usage

```shell
python3 dataset_processor.py <folder_containing_tests>
```

Where `<folder_containing_tests>` is the top most folder here:

```
my_tests
    600SEC_167B_12P_4S_BE_UC_3DUR_100LC
    600SEC_12334412B_22P_12S_REL_MC_0DUR_100LC
    ...
```

# Introduction
What do we want to this to do?
- [ ] Gather the test data from each test
  - [ ] Record the parameter configuration values
  - [ ] Record the distribution recreation stats for each metric
- [ ] For each test:
  - [ ] Get the distribution of data for each metric.
  - [ ] Cut off the first 20% to be conservative about transient removal.
  - [ ] Calculate the distribution recreation stats for each metric.
- [ ] Store it all into one single dataframe and write it to a single spreadsheet which will be the final dataset.