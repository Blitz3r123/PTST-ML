import warnings
import unittest
import datetime
import os
import a01_results_analysis as src
import pandas as pd
import numpy as np

def generate_random_result_row(
        setting=None,
        model_type=None,
        int_or_ext=None,
        metric_of_interest=None,
    ):
        mdoel_type = model_type if model_type else np.random.choice([
            'Linear Regression', 
            'Random Forests'
        ])
        int_or_ext = int_or_ext if int_or_ext else np.random.choice([
            'interpolation', 
            'extrapolation'
        ])
        train_dataset = np.random.choice([
            'PCG + RCG (Inclusive)', 
            'PCG + RCG (Exclusive)'
        ])
        train_dataset_filename = "train.csv"
        test_dataset = np.random.choice([
            'PCG + RCG (Inclusive)', 
            'PCG + RCG (Exclusive)'
        ])
        test_dataset_filename = "test.csv"
        train_example_count = np.random.randint(800, 1000)
        test_example_count = np.random.randint(10, 200)
        input_variables = "datalen_bytes, pub_count, sub_count, reliability, multicast, durability_0, durability_1, durability_2, durability_3"
        metric_of_interest = metric_of_interest if metric_of_interest else np.random.choice([
            'latency_us',
            'total_throughput_mbps',
            'avg_throughput_mbps',
            'total_samples_per_sec',
            'avg_samples_per_sec',
            'total_lost_samples',
            'avg_lost_samples',
            'total_lost_samples_percentage',
            'avg_lost_samples_percentage',
            'total_received_samples',
            'avg_received_samples',
            'total_received_samples_percentage',
            'avg_received_samples_percentage',
        ])
        percentile = np.random.choice([
            'min', 'max',
            'mean', 'std',
            '1', '2', '5', '10', 
            '25', '30', '40', '50', '60', '70', '75', '80', 
            '90', '95', '99'
        ])
        output_variable = f"{metric_of_interest}_{percentile}"
        standardisation_function = np.random.choice([
            "none", 
            "z_score", 
            'min_max', 
            'robust_scaler'
        ])
        transform_function = np.random.choice([
            "none",
            "log",
            "log10",
            "log2",
            "log1p",
            "sqrt",
        ])
        error_type = np.random.choice([
            "rmse", 
            "mse", 
            "mae", 
            "mape", 
            "r2", 
            "medae", 
            "explained_variance"
        ])
        train_error = np.random.rand()
        test_error = np.random.rand()

        if setting:
            if "nantrain" in setting:
                train_error = np.nan
            if "nantest" in setting:
                test_error = np.nan
            if "inftrain" in setting:
                train_error = np.inf
            if "inftest" in setting:
                test_error = np.inf

        return {
            'model_type': model_type,
            'created_at': datetime.datetime.now(),
            'int_or_ext': int_or_ext,
            'train_dataset': train_dataset,
            'train_dataset_filename': train_dataset_filename,
            'test_dataset': test_dataset,
            'test_dataset_filename': test_dataset_filename,
            'train_example_count': train_example_count,
            'test_example_count': test_example_count,
            'input_variables': input_variables,
            'output_variable': output_variable,
            'metric_of_interest': metric_of_interest,
            'standardisation_function': standardisation_function,
            'transform_function': transform_function,
            'error_type': error_type,
            'train_error': train_error,
            'test_error': test_error
        }

class TestA01ResultsAnalysis(unittest.TestCase):

    def setUp(self):
        test_files = [
            "2021-02-03_16-48-03_results.csv",
            "2022-02-07_16-48-03_results.csv",
            "2022-01-07_16-48-03_results.csv",
        ]

        for file in test_files:
            with open(file, "w") as f:
                f.write("")

        result_columns = [
            'model_type',
            'created_at',
            'int_or_ext',
            'train_dataset',
            'train_dataset_filename',
            'test_dataset',
            'test_dataset_filename',
            'train_example_count',
            'test_example_count',
            'input_variables',
            'output_variable',
            'metric_of_interest',
            'standardisation_function',
            'transform_function',
            'error_type',
            'train_error',
            'test_error'
        ]

        test_result_df = pd.DataFrame(columns=result_columns, data=[generate_random_result_row()])

        for model_type in ['Linear Regression', 'Random Forests']:
            for int_or_ext in ['interpolation', 'extrapolation']:
                for metric_of_interest in [
                    'latency_us',
                    'total_throughput_mbps',
                    'avg_throughput_mbps',
                    'total_samples_per_sec',
                    'avg_samples_per_sec',
                    'total_lost_samples',
                    'avg_lost_samples',
                    'total_lost_samples_percentage',
                    'avg_lost_samples_percentage',
                    'total_received_samples',
                    'avg_received_samples',
                    'total_received_samples_percentage',
                    'avg_received_samples_percentage',
                ]:
                    for i in range(10):
                        test_result_df = pd.concat([
                            test_result_df, 
                            pd.DataFrame(columns=result_columns, data=[generate_random_result_row(
                                model_type=model_type,
                                int_or_ext=int_or_ext,
                                metric_of_interest=metric_of_interest
                            )])
                        ], ignore_index=True)

        test_result_df.to_csv("2021-02-03_16-48-03_test-results.csv", index=False)

    def tearDown(self):
        test_files = [
            "2021-02-03_16-48-03_results.csv",
            "2022-02-07_16-48-03_results.csv",
            "2022-01-07_16-48-03_results.csv",
        ]

        for file in test_files:
            os.remove(file)

        os.remove("2021-02-03_16-48-03_test-results.csv")

    def test_get_timestamps_from_filenames(self):
        test_csv_files1 = [
            '2024-02-03_16-48-03_results.csv',
            '2024-02-07_16-48-03_results.csv',
            '2024-01-07_16-48-03_results.csv',
        ]

        self.assertEqual(
            src.get_timestamps_from_filenames(test_csv_files1), 
            [
                datetime.datetime(2024, 2, 3, 16, 48, 3),
                datetime.datetime(2024, 2, 7, 16, 48, 3),
                datetime.datetime(2024, 1, 7, 16, 48, 3)
            ]
        )

        test_csv_files2 = [
            '2024-02-03_16-48-03_results.csv',
            '2024-02-07_16-48-03_results.csv',
            '2024-01-07_16-48-03.csv',
        ]

        self.assertEqual(
            src.get_timestamps_from_filenames(test_csv_files2), 
            [
                datetime.datetime(2024, 2, 3, 16, 48, 3),
                datetime.datetime(2024, 2, 7, 16, 48, 3)
            ]
        )

        test_csv_files3 = [
            'something.csv',
            'something_else.csv',
        ]

        self.assertEqual(
            src.get_timestamps_from_filenames(test_csv_files3), 
            []
        )

    def test_get_latest_result_file(self):
        
        test_csv_files = [
            '2024-02-03_16-48-03_results.csv',
            '2024-02-07_16-48-03_results.csv',
            '2024-01-07_16-48-03_results.csv',
        ]

        self.assertEqual(
            src.get_latest_result_file([]), 
            None
        )

        self.assertEqual(
            src.get_latest_result_file(["results.csv"]), 
            "results.csv"
        )
        
        self.assertEqual(
            src.get_latest_result_file([test_csv_files[0]]), 
            "2024-02-03_16-48-03_results.csv"
        )

        self.assertEqual(
            src.get_latest_result_file(test_csv_files), 
            "2024-02-07_16-48-03_results.csv"
        )

    def test_get_model_results(self):
        df = src.get_model_results()
        
        self.assertEqual(
            type(df), 
            type(pd.DataFrame())
        )

        self.assertGreaterEqual(
            len(df), 
            0
        )

    def test_is_df_valid(self):
        df = src.get_model_results()
        
        self.assertEqual(src.is_df_valid(df), True)
        self.assertEqual(src.is_df_valid(pd.DataFrame()), False)
        self.assertEqual(src.is_df_valid(None), False)

    def test_get_table_columns(self):
        self.assertEqual(src.get_table_columns([]), [])
        self.assertEqual(src.get_table_columns(['']), [])
        self.assertEqual(src.get_table_columns(['2']), [])
        self.assertEqual(src.get_table_columns(['2', 'rmse']), ['RMSE Train', 'RMSE Test'])
        self.assertEqual(src.get_table_columns(['  ', 'rmse']), ['RMSE Train', 'RMSE Test'])
        self.assertEqual(src.get_table_columns(['train']), [])
        self.assertEqual(src.get_table_columns(['test']), [])

    def test_format_stats(self):
        self.assertEqual(src.format_stats([]), [])
        self.assertEqual(src.format_stats(['']), [])
        self.assertEqual(src.format_stats(['1']), ['1st'])
        self.assertEqual(src.format_stats(['2']), ['2nd'])
        self.assertEqual(src.format_stats(['3']), ['3rd'])
        self.assertEqual(src.format_stats(['4']), ['4th'])
        self.assertEqual(src.format_stats(['mean']), ['Mean'])
        self.assertEqual(src.format_stats(['std']), ['std'])
        self.assertEqual(src.format_stats(['mean', 'std', '1', '2', '5', '10']), ['Mean', 'std', '1st', '2nd', '5th', '10th'])

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=FutureWarning)
    unittest.main()