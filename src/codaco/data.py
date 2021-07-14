import pandas as pd
import numpy as np
import pathlib

# TODO: function for loading ML-datasets as generators

def load_dataset(identifier, source="file", download_to="datasets"):
    """
    Main function to download datasets from different sources.
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
