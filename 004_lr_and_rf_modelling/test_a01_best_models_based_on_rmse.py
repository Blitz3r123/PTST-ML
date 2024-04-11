import warnings
import unittest
import a01_best_models_based_on_rmse as src
import pandas as pd
import numpy as np
from icecream import ic

from constants import *

class TestA01BestModelsBasedOnRmse(unittest.TestCase):
    def setUp(self):
        pass
        
    def tearDown(self):
        pass

    def test_get_results_df(self):
        df = src.get_results_df()
        self.assertIsNotNone(df)

        wanted_cols = [
            'model_type',
            'created_at',
            'int_or_ext',
            'train_dataset',
            'train_dataset_filename',
            'test_dataset',
            'test_dataset_filename',
            'train_example_count',
            'test_example_count',
            'input_variable',
            'output_variable',
            'metric_of_interest',
            'standardisation_function',
            'transform_function',
            'r2_train_error',
            'r2_test_error',
            'rmse_train_error',
            'rmse_test_error'
        ]
        self.assertEqual(
            len(
                list(
                    set(df.columns) - set(wanted_cols)
                )
            ) >= 0,
            True
        )

    def test_get_csv_files(self):
        csv_files = src.get_csv_files('this_folder_doesnt_exist')
        self.assertEqual(csv_files, [])

    def test_calculate_average_metrics_from_model(self):
        func_return_value = src.calculate_average_metrics_from_models(None)
        self.assertEqual(func_return_value, None)

        test_df = pd.DataFrame()
        func_return_value = src.calculate_average_metrics_from_models(test_df)
        self.assertEqual(func_return_value, None)

        test_df = pd.DataFrame(
            [
                {'random_forest', 'interpolation'}
            ],
            columns=['model_type', 'int_or_ext']
        )
        func_return_value = src.calculate_average_metrics_from_models(test_df)
        self.assertEqual(func_return_value, None)

        test_df = pd.DataFrame([], columns = [
            'model_type',
            'int_or_ext',
            'standardisation_function', 
            'transform_function',
            'r2_test_error',
            'rmse_test_error'
        ])
        func_return_value = src.calculate_average_metrics_from_models(test_df)
        # todo check that the df is normal

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=FutureWarning)
    unittest.main()
