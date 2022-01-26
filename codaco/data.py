import pandas as pd
import numpy as np
import pathlib
import requests
import re
import warnings
import shutil
import subprocess

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


def extract_zip(filepath: pathlib.Path, outdir: pathlib.Path):
    known = list(outdir.iterdir())
    # extract zip files
    try:
        shutil.unpack_archive(filepath, outdir)
        filepath.unlink()
    except shutil.ReadError as e:
        pass
    ret = subprocess.run(["uncompress", str(filepath.absolute())], cwd=outdir, capture_output=True)
    if ret.returncode != 0:
        raise IOError(f"could not extract {filepath.absolute()}")
    # check if extracted files need to be extracted again
    for f in (x for x in outdir.iterdir() if x not in known and x.suffix in [".zip", ".gz", ".tar", ".bz2", ".Z"]):
        extract_zip(f, outdir)


def load_ucimlr(identifier, download_to="datasets"):
    exclude = [
        "artificial-characters",
        "audiology",
        "chess/king-rook-vs-king-knight",
        "chess/domain-theories",
        "chorales",
        "diabetes",
        "dgp-2",
        "document-understanding",
        "ebl"
    ]
    if identifier in exclude:
        return None
    url = f"https://archive.ics.uci.edu/ml/machine-learning-databases/{identifier}/"
    refs = re.findall(r"href=\"(.+?)\"", requests.get(url).text)
    refs = [x for x in refs if not x.startswith('/') and x not in ["Index"]]
    print(refs)
    outdir = pathlib.Path(download_to).joinpath(identifier)
    outdir.mkdir(parents=True, exist_ok=True)
    for f in refs:
        if f == "Index":
            continue
        outfile = outdir.joinpath(f)
        if not outfile.exists():
            download_file(url + f, outfile)
            if outfile.suffix == '.Z':
                # extract zip files
                extract_zip(outfile, outdir)

    namefiles = [x for x in outdir.iterdir() if ".names" in x.suffixes]
    datafiles = [x for x in outdir.iterdir() if ".data" in x.suffixes]
    if len(datafiles) == 0:
        raise IOError(f"Could not find data file for UCIMLR database {identifier}, I do not know how to load this dataset. :(")
    elif len(datafiles) > 1:
        warnings.warn(f"Found multiple datafiles for UCIMLR database {identifier}: {datafiles}")
    columns = None if len(namefiles) == 0 else guess_ucimlr_columns(namefiles[0])
    return pd.read_csv(datafiles[0], names=columns, on_bad_lines="warn", encoding_errors="backslashreplace")

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
