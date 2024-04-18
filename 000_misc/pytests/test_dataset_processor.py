import unittest
import warnings

import dataset_processor as dp

class TestDatasetProcessor(unittest.TestCase):

    def test_main(self):
        pass
        # self.assertEqual(dp.main(), False)
        # self.assertEqual(dp.main([]), False)
        # self.assertEqual(dp.main(['']), False)
        # self.assertEqual(dp.main(['something_that_doesnt_exist']), False)
        # self.assertEqual(dp.main(['pytests/test_data']), True)
        # self.assertEqual(dp.main(['pytests/test_data/empty_test_data_folder']), False)
        # self.assertEqual(dp.main(['pytests/test_data/empty_test_data_folder']), False)
        # self.assertEqual(dp.main(['pytests/test_data/test_data_one']), False)
        # self.assertEqual(dp.main(['pytests/test_data/test_data_two']), False)

    def test_get_longest_path_in_dir(self):
        self.assertEqual(
            dp.get_longest_path_in_dir(''),
            ""
        )
        self.assertEqual(
            dp.get_longest_path_in_dir('pytests/test_data'), 
            'pytests/test_data/test_data_two/a_folder_with_this_name/data.csv'
        )
        self.assertEqual(
            dp.get_longest_path_in_dir('pytests/test_data/empty_test_data_folder'), 
            ""
        )
        self.assertEqual(
            dp.get_longest_path_in_dir('pytests/test_data/test_data_one'),
            ""
        )
        self.assertEqual(
            dp.get_longest_path_in_dir('pytests/test_data/test_data_two'), 
            "pytests/test_data/test_data_two/a_folder_with_this_name/data.csv"
        )

    def test_get_test_parent_dirpath_from_fullpath(self):
        self.assertEqual(
            dp.get_test_parent_dirpath_from_fullpath(
                "phd/year_one/machine_learning/experiments/qos_capture/600SEC_200B/pub.csv"
            ),
            "phd/year_one/machine_learning/experiments/qos_capture"
        )

        self.assertEqual(
            dp.get_test_parent_dirpath_from_fullpath(""),
            None
        )

        self.assertEqual(
            dp.get_test_parent_dirpath_from_fullpath(),
            None
        )

        self.assertEqual(
            dp.get_test_parent_dirpath_from_fullpath("phd/qos_capture"),
            "phd/qos_capture"
        )

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=FutureWarning)
    unittest.main()
