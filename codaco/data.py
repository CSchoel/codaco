import pandas as pd
import numpy as np
import pathlib
import requests
import re

# TODO: function for loading ML-datasets as generators

def load_dataset(identifier, source="file", download_to="datasets"):
    """
    Main function to download datasets from different sources.

    Note that this function is designed to be as simple as possible, if you are missing
    any arguments that you want to pass, you probably want to call a more specific
    function like pandas.read_csv instead.

    Currently, the following values are allowed for the source parameter:

    - "file": loads data from a single file on the local file system
    - "ucimlr": loads data from the UCI machine learning repository

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
        raise IOError("Unknown dataset source: {}".format(source))


def load_file(fname):
    """
    Loads a dataset from a file, detecting the file type based on the extension.
    """

    if fname.suffix == "csv":
        return pd.read_csv(fname)
    else:
        raise "Sorry, I cannot load .{} files.".format(fname.suffix)

def load_ucimlr(identifier, download_to="datasets"):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/{}/".format(identifier)
    index = requests.get(url + "Index").text
    # Index format example
    # 02 Dec 1996      139 Index
    # 02 Mar 1993    11903 glass.data
    # 16 Jul 1992      780 glass.tag
    # 30 May 1989     3506 glass.names
    files = re.findall(r"\d+\s\w+\s\d+\s+\d+\s+(.*)", index)
    outdir = pathlib.Path(download_to).joinpath(identifier)
    outdir.mkdir(parents=True, exist_ok=True)
    for f in files:
        if f == "Index":
            continue
        outfile = outdir.joinpath(f)
        if not outfile.exists():
            download_file(url + f, outfile)
    namefile = outdir.joinpath("{}.names".format(identifier))
    columns = guess_ucimlr_columns(namefile)
    datafile = outdir.joinpath("{}.data".format(identifier))
    if datafile.name in files:
        return pd.read_csv(datafile, names=columns)
    else:
        raise IOError("Could not find file named {}, I do not know how to load this dataset. :(".format(datafile.name))

def guess_ucimlr_columns(namefile):
    if not namefile.exists():
        return None
    text = namefile.read_text(encoding="utf-8")
    return None

def download_file(url, path):
    with requests.get(url, stream=True) as r:
        # raise error if HTTP error code was returned
        r.raise_for_status()
        with path.open(mode="wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)
