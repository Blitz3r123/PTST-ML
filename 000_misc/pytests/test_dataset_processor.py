import unittest
import warnings
import pandas as pd
import dataset_processor as dp

from icecream import ic

class TestDatasetProcessor(unittest.TestCase):

    def test_get_longest_path_in_dir(self):
        self.assertEqual(
            dp.get_longest_path_in_dir(""),
            None
        )
        self.assertEqual(
            dp.get_longest_path_in_dir(),
            None
        )
        self.assertEqual(
            dp.get_longest_path_in_dir('pytests/test_data'), 
            'pytests/test_data/normal_tests/600SEC_2344B_24P_12S_BE_UC_1DUR_100LC/p1_eicmp6.log'
        )
        self.assertEqual(
            dp.get_longest_path_in_dir('pytests/test_data/empty_test_data_folder'), 
            ""
        )
        self.assertEqual(
            dp.get_longest_path_in_dir('pytests/test_data/test_data_one'),
            "pytests/test_data/test_data_one/.DS_Store"
        )
        self.assertEqual(
            dp.get_longest_path_in_dir('pytests/test_data/test_data_two'), 
            "pytests/test_data/test_data_two/a_folder_with_this_name/an_extra_folder/data.csv"
        )

    def test_get_test_parent_dirpath_from_fullpath(self):
        # Normal Case
        self.assertEqual(
            dp.get_test_parent_dirpath_from_fullpath(
                "phd/year_one/machine_learning/experiments/qos_capture/600SEC_200B/pub.csv"
            ),
            "phd/year_one/machine_learning/experiments/qos_capture"
        )

        # Empty Case
        self.assertEqual(
            dp.get_test_parent_dirpath_from_fullpath(""),
            None
        )

        # Empty Case
        self.assertEqual(
            dp.get_test_parent_dirpath_from_fullpath(),
            None
        )

        # Edge Case
        self.assertEqual(
            dp.get_test_parent_dirpath_from_fullpath("phd/qos_capture"),
            None
        )

        self.assertEqual(
            dp.get_test_parent_dirpath_from_fullpath(
                "phd/"
            ),
            None
        )

    def test_get_headings_from_csv_file(self):
        self.assertEqual(
            dp.get_headings_from_csv_file(
                'pytests/test_data/normal_tests/600SEC_2241B_6P_20S_BE_UC_1DUR_100LC/pub_0.csv'
            ),
            [
                'Length (Bytes)',
                'Latency (μs)',
                'Ave (μs)',
                'Std (μs)',
                'Min (μs)',
                'Max (μs)'
            ]
        )

        self.assertEqual(
            dp.get_headings_from_csv_file(
                'pytests/test_data/normal_tests/600SEC_2121B_6P_20S_BE_UC_1DUR_100LC/pub_0.csv'
            ),
            [
                'Length (Bytes)',
                'Latency (μs)',
            ]
        )

        self.assertEqual(
            dp.get_headings_from_csv_file(
                None
            ),
            None
        )

        self.assertEqual(
            dp.get_headings_from_csv_file(
                'some_file_that/does_not/exist.csv'
            ),
            None
        )

        sub_filepath = "pytests/test_data/normal_tests/600SEC_212B_6P_20S_BE_UC_1DUR_100LC/sub_0.csv"

        sub_headings = dp.get_headings_from_csv_file(sub_filepath)
        self.assertEqual(
            sub_headings,
            [
                'Length (Bytes)',
                'Total Samples',
                'Samples/s',
                'Avg Samples/s',
                'Mbps',
                'Avg Mbps',
                'Lost Samples',
                'Lost Samples (%)'
            ]
        )

    def test_get_latency_df_from_testdir(self):
        self.assertEqual(
            type(
                dp.get_latency_df_from_testdir(
                    "pytests/test_data/normal_tests/600SEC_2241B_6P_20S_BE_UC_1DUR_100LC/"
                )
            ),
            pd.Series
        )

        self.assertEqual(
            type(
                dp.get_latency_df_from_testdir(
                    "pytests/test_data/normal_tests/600SEC_2241B_6P_20S_BE_UC_1DUR_100LC/"
                )
            ),
            pd.Series
        )

        df = dp.get_latency_df_from_testdir(
            "pytests/test_data/normal_tests/600SEC_1B_1P_1S_BE_UC_1DUR_1LC/"
        )
        self.assertEqual(df.mean(), 1)
        self.assertEqual(len(df.index), 9)

    def test_get_sub_files_from_testdir(self):
        sub_files = dp.get_sub_files_from_testdir(
            "pytests/test_data/normal_tests/600SEC_2241B_6P_20S_BE_UC_1DUR_100LC/"
        )

        self.assertEqual(len(sub_files), 20)

        self.assertEqual(
            dp.get_sub_files_from_testdir(
                "pytests/test_data/normal_tests/some_random_folder"
            ),
            None
        )

        self.assertEqual(
            len(
                dp.get_sub_files_from_testdir(
                    "pytests/test_data/normal_tests/600SEC_23P_1S_REL_MC_0DUR_100LC/"
                )
            ),
            4
        )

        self.assertEqual(
            dp.get_sub_files_from_testdir(
                "idk"
            ),
            None
        )

        self.assertEqual(
            dp.get_sub_files_from_testdir(),
            None
        )

        self.assertEqual(
            dp.get_sub_files_from_testdir("pytests"),
            []
        )

    def test_get_sub_metric_df_from_testdir(self):
        df = dp.get_sub_metric_df_from_testdir(
            "pytests/test_data/normal_tests/600SEC_2241B_6P_20S_BE_UC_1DUR_100LC/",
            "mbps"
        )

        self.assertEqual(
            type(df),
            pd.DataFrame
        )

    def test_get_test_param_df_from_testdir(self):
        param_df = dp.get_test_param_df_from_testdir(
            "pytests/test_data/normal_tests/600SEC_2241B_6P_20S_BE_UC_1DUR_100LC/"
        )
        self.assertEqual(param_df.iloc[0]['duration_sec'], 600)
        self.assertEqual(param_df.iloc[0]['datalen_byte'], 2241)
        self.assertEqual(param_df.iloc[0]['pub_count'], 6)
        self.assertEqual(param_df.iloc[0]['sub_count'], 20)
        self.assertEqual(param_df.iloc[0]['use_reliable'], 0)
        self.assertEqual(param_df.iloc[0]['use_multicast'], 0)
        self.assertEqual(param_df.iloc[0]['durability'], 1)
        self.assertEqual(param_df.iloc[0]['latency_count'], 100)

        param_df = dp.get_test_param_df_from_testdir(
            "pytests/test_data/normal_tests/600S_2241B_6P_20S_BE_UC_1DUR_100LC/"
        )
        self.assertEqual(param_df.iloc[0]['duration_sec'], 600)
        self.assertEqual(param_df.iloc[0]['datalen_byte'], 2241)
        self.assertEqual(param_df.iloc[0]['pub_count'], 6)
        self.assertEqual(param_df.iloc[0]['sub_count'], 20)
        self.assertEqual(param_df.iloc[0]['use_reliable'], 0)
        self.assertEqual(param_df.iloc[0]['use_multicast'], 0)
        self.assertEqual(param_df.iloc[0]['durability'], 1)
        self.assertEqual(param_df.iloc[0]['latency_count'], 100)


        param_df = dp.get_test_param_df_from_testdir("")
        self.assertEqual(param_df, None)

        param_df = dp.get_test_param_df_from_testdir(
            "pytests/test_data/normal_tests/600SEC_23P_1S_REL_MC_0DUR_100LC/"
        )
        self.assertEqual(param_df, None)

        self.assertEqual(
            dp.get_test_param_df_from_testdir(
                "pytests/test_data/normal_tests/some_random_folder/"
            ),
            None
        )
        self.assertEqual(
            dp.get_test_param_df_from_testdir(
                "pytests/test_data/normal_tests/600SEC_2241B_6P_S_BE_UC_1DUR_100LC/"
            ),
            None
        )

    def test_get_pub_file_from_testdir(self):
        self.assertEqual(
            dp.get_pub_file_from_testdir("pytests/test_data/normal_tests/600SEC_2241B_6P_20S_BE_UC_1DUR_100LC/"),
            "pytests/test_data/normal_tests/600SEC_2241B_6P_20S_BE_UC_1DUR_100LC/pub_0.csv"
        )

        self.assertEqual(
            dp.get_pub_file_from_testdir(
                "pytests/test_data/normal_tests/600SEC_2121B_6P_20S_BE_UC_1DUR_100LC/"
            ),
            "pytests/test_data/normal_tests/600SEC_2121B_6P_20S_BE_UC_1DUR_100LC/pub_0.csv"
        )

        self.assertEqual(
            dp.get_pub_file_from_testdir(
                "pytests/test_data/normal_tests/600SEC_212B_6P_20S_BE_UC_1DUR_100LC/"
            ),
            None
        )

        self.assertEqual(
            dp.get_pub_file_from_testdir(
                "some_random/file_path/that_does_not/exist"
            ),
            None
        )

    def test_get_test_name_from_test_dir(self):
        # Normal Case
        test_name = dp.get_test_name_from_test_dir(
            "pytests/test_data/normal_tests/600SEC_2241B_6P_20S_BE_UC_1DUR_100LC/"
        )
        self.assertEqual(
            test_name,
            "600SEC_2241B_6P_20S_BE_UC_1DUR_100LC"
        )

        # Empty Case
        self.assertEqual(
            dp.get_test_name_from_test_dir(""),
            None
        )

        self.assertEqual(
            dp.get_test_name_from_test_dir(),
            None
        )

        self.assertEqual(
            dp.get_test_name_from_test_dir(
                "600SEC_2241B_6P_20S_BE_UC_1DUR_100LC"
            ),
            "600SEC_2241B_6P_20S_BE_UC_1DUR_100LC"
        )

    def test_get_distribution_stats_from_col(self):
        df = dp.get_latency_df_from_testdir(
            "pytests/test_data/normal_tests/600SEC_1B_1P_1S_BE_UC_1DUR_1LC/"
        )

        expected_dist_stats = {
            'mean': 1,
            'std': 0,
            'min': 1,
            'max': 1,
            '1%': 1,
            '2%': 1,
            '5%': 1,
            '10%': 1,
            '20%': 1,
            '25%': 1,
            '30%': 1,
            '40%': 1,
            '50%': 1,
            '60%': 1,
            '70%': 1,
            '75%': 1,
            '80%': 1,
            '90%': 1,
            '95%': 1,
            '98%': 1,
            '99%': 1,
        }
        actual_dist_stats = dp.get_distribution_stats_from_col(df)

        self.assertEqual(
            expected_dist_stats,
            actual_dist_stats
        )

    def test_get_distribution_stats_df(self):
        throughput_df = dp.get_sub_metric_df_from_testdir(
            "pytests/test_data/normal_tests/600SEC_2241B_6P_20S_BE_UC_1DUR_100LC/",
            'mbps'
        )
        
        dist_stat_df = dp.get_distribution_stats_df(
            throughput_df
        )

        self.assertEqual(type(dist_stat_df), pd.DataFrame)

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=FutureWarning)
    unittest.main()
