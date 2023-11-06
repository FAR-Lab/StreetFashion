# FARLAB - UrbanECG Project
# Developer: @mattwfranchi, with help from GitHub Copilot
# Last Edited: 11/05/2023

# This file contains a function decorator to save the output of a function to a csv file.

# Import the necessary packages
import os
import sys
import pandas as pd 
import logging 
import time
import datetime
import functools

sys.path.append(os.path.join(".."))
sys.path.append(os.path.join("..", ".."))

from user.params.io import PALETTE_OUTPUT_DIR_PRE_AGG

def to_csv(prefix): 
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            # make sure the output directory exists
            os.makedirs(f"{PALETTE_OUTPUT_DIR_PRE_AGG}/{prefix}", exist_ok=True)

            now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            result = func(*args, **kwargs)
            if isinstance(result, pd.DataFrame):
                result.to_csv(f'{PALETTE_OUTPUT_DIR_PRE_AGG}/{prefix}/{now}_{func.__name__}.csv', index=False)
            else:
                logging.error(f'Function {func.__name__!r} did not return a pandas DataFrame.')
            return func(*args, **kwargs)
        return wrapper
    return decorator

