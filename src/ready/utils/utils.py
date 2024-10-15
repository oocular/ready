"""
utils
"""

import os
from pathlib import Path

HOME_PATH = Path.home()
REPOSITORY_PATH = Path.cwd()



def set_data_directory(main_path: str = None, data_path: str = None):
    """
    set_data_directory with input variable:
        data_path.
        For example:
        set_data_directory("data/mobious/sample-frames/test640x400")
    """
    if main_path is None:
        main_path = REPOSITORY_PATH
    os.chdir(os.path.join(main_path, data_path))
