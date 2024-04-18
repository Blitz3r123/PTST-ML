#==========================================================================================================================
#  ?                                                     ABOUT
#  @author         :    Kaleem Peeroo
#  @email          :    Kaleem.Peeroo@city.ac.uk 
#  @repo           :    https://github.com/Blitz3r123/PTST-ML 
#  @createdOn      :    2024-02-20 
#  @description    :    Take raw test data and put into a single spreadsheet. 
#==========================================================================================================================

import os
import sys
import pytest
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#--------------------------------------------------------------------------------------------------------------------------
#                                                      FUNCTIONS
#--------------------------------------------------------------------------------------------------------------------------
def get_longest_path_in_dir(dir_path):
    """
    Get the longest path in a directory.
    """
    longest_path = ""
    longest_path_len = 0

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            if len(file_path) > longest_path_len:
                longest_path = file_path
                longest_path_len = len(file_path)
    
    return longest_path

def get_test_parent_dirpath_from_fullpath(longest_path=""):
    if "/" not in longest_path:
        logger.error(f"No / found in {longest_path}.")
        return None

    longest_path_items = longest_path.split("/")
    if len(longest_path_items) <= 2:
        return longest_path

    return "/".join(longest_path_items[:-2])

def main(sys_args=None):
    if not sys_args:
        logger.error("No sys args provided.")
        return False
    
    if len(sys_args) != 1:
        logger.error(f"Only one sys arg is allowed. You have provided {len(sys_args)}: {sys_args}")
        return False
    
    if not os.path.exists(sys_args[0]):
        logger.error("File does not exist.")
        return False

    if not os.path.isdir(sys_args[0]):
        logger.error("File is not a directory.")
        return False

    if not os.listdir(sys_args[0]):
        logger.error("Directory is empty.")
        return False
    
    tests_dir_path = sys_args[0]

    # tests_dir_path should be in the format: dir_path/600SEC.../pub0.csv
    # A single test should be in the final folder e.g. 600SEC.../
    # So get the longest path and then go out 2 folders to see all tests.
    # e.g. my_path/some_path/more_folders/600SEC.../pub0.csv
    #   => my_path/some_path/more_folders/

    longest_path = get_longest_path_in_dir(tests_dir_path)

    test_parent_dirpath = get_test_parent_dirpath_from_fullpath(longest_path)


if __name__ == "__main__":
    if pytest.main(["-q", "./pytests", "--exitfirst"]) == 0:
        main(sys.argv[1:])
    else:
        logger.error("Tests failed.")
        sys.exit(1)
