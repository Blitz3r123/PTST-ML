import unittest
import warnings
import pandas as pd

import dataset_processor as dp

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

    def test_get_headings_from_pub_file(self):
        self.assertEqual(
            dp.get_headings_from_pub_file(
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
            dp.get_headings_from_pub_file(
                'pytests/test_data/normal_tests/600SEC_2121B_6P_20S_BE_UC_1DUR_100LC/pub_0.csv'
            ),
            [
                'Length (Bytes)',
                'Latency (μs)',
            ]
        )

        self.assertEqual(
            dp.get_headings_from_pub_file(
                None
            ),
            []
        )

        self.assertEqual(
            dp.get_headings_from_pub_file(
                'some_file_that/does_not/exist.csv'
            ),
            []
        )

    def test_get_latency_df_from_testdir(self):
        self.assertEqual(
            type(
                dp.get_latency_df_from_testdir(
                    "pytests/test_data/normal_tests/600SEC_2241B_6P_20S_BE_UC_1DUR_100LC/"
                )
            ),
            pd.DataFrame
        )

        self.assertEqual(
            type(
                dp.get_latency_df_from_testdir(
                    "pytests/test_data/normal_tests/600SEC_2241B_6P_20S_BE_UC_1DUR_100LC/"
                )
            ),
            pd.DataFrame
        )

    def test_get_sub_files_from_testdir(self):
        sub_files = dp.get_sub_files_from_testdir(
            "pytests/test_data/normal_tests/600SEC_2241B_6P_20S_BE_UC_1DUR_100LC/"
        )

        self.assertEqual(len(sub_files), 20)

    def test_get_sub_headings_from_sub_file(self):
        sub_filepath = "pytests/test_data/normal_tests/600SEC_212B_6P_20S_BE_UC_1DUR_100LC/sub_0.csv"

        sub_headings = dp.get_headings_from_sub_file(sub_filepath)
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


    def test_get_distribution_stats_from_col(self):
        # TODO: this
        pass

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
