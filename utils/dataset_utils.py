import pandas as pd
import os

from pathlib import Path

def save_to_csv(path, df, index):
    """
    Save the pd.DataFrame as csv file in path
    :param path: path of the file
    :param df: dataframe to save
    :param index: True or False (if index in csv ir not)
    """
    p = Path(path)
    if not os.path.exists(p.parent):
        os.makedirs(p.parent)
    df.to_csv(path_or_buf=path, index=index)


def load_from_csv(path):
    """
    Load a csv datasets as a pd.DataFrame
    :param path: path of the csv file
    :return: pd.DataFrame
    """
    return pd.read_csv(path)













