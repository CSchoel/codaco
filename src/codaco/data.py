import pandas as pd
import numpy as np
import pathlib

# TODO: function for loading ML-datasets as generators

def load_dataset(identifier, source="file", download_to="datasets"):
    """
    Main function to download datasets from different sources.

    Currently, the following values are allowed for the source parameter:

    - "file": loads data from a single file on the local file system
    - "ucimlr": loads data from the UCI machine learning repository (using the ucimlr package)

    Args:
        identifier (str): unique identifier of the dataset (file name, database number, ...)

    Kwargs:
        source (str): source from which to load the data
        download_to (str): folder where downloaded data should be stored

    Returns:
        pandas dataframe
    """

    if source == "file":
        return load_file(pathlib.Path(identifier))
    elif source == "ucimlr":
        return load_ucimlr(identifier, download_to=download_to)
    else:
        raise "Unknown dataset source: {}".format(source)


def load_file(fname):
    """
    Loads a dataset from a file, detecting the file type based on the extension.
    """

    if fname.suffix == "csv":
        return pd.read_csv(fname)
    else:
        raise "Sorry, I cannot load .{} files.".format(fname.suffix)

def load_ucimlr(identifier, download_to="datasets"):
    pass
