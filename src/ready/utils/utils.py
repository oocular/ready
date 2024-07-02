"""
utils modules
"""

import os


def get_working_directory():
    """
    Test working directory
    """

    pwd = os.getcwd()
    if os.path.split(pwd)[-1] == "openeds":
        os.chdir("../../../../")  # for segnet
    else:
        os.chdir("..")  # for tests
    get_pwd = os.path.abspath(os.getcwd())
    return get_pwd


def set_data_directory(global_data_path: str):
    """
    set_data_directory with input variable:
        global_data_path. For example: "datasets/openEDS"
    """
    os.chdir(os.path.join(get_working_directory(), global_data_path))
