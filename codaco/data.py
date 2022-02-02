import pandas as pd
import numpy as np
from pathlib import Path
import requests
from urllib.parse import urlparse, ParseResult, urljoin
import re
import warnings
import shutil
import subprocess
from typing import *

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
        return load_file(Path(identifier))
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


def extract_zip(filepath: Path, outdir: Path):
    known = list(outdir.iterdir())
    # extract zip files
    try:
        shutil.unpack_archive(filepath, outdir)
        filepath.unlink()
    except shutil.ReadError as e:
        ret = subprocess.run(["uncompress", "-f", str(filepath.absolute())], cwd=outdir, capture_output=True)
        if ret.returncode != 0:
            raise IOError(f"could not extract {filepath.absolute()}")
        # check if extracted files need to be extracted again
    for f in (x for x in outdir.iterdir() if x not in known and x.suffix in [".zip", ".gz", ".tar", ".bz2", ".Z"]):
        extract_zip(f, outdir)

def download_recursive(url: str, outdir: Path, parents: bool=False, exclude_html: bool=True) -> List[Path]:
    outdir.mkdir(exist_ok=True, parents=True)
    fname = outdir / Path(urlparse(url).path).name
    if fname.exists():
        return []
    downloaded = []
    req = requests.get(url)
    if "HTTP" in req.headers['content-type']:
        # download recursive
        text = req.text
        refs = re.findall(r"href=\"(.+?)\"", text)
        for r in refs:
            refurl = urljoin(url, r)
            if url.startswith(refurl) and not parents:
                continue
            downloaded += download_recursive(
                refurl,
                outdir / Path(urlparse(refurl).path).parent.relative_to(urlparse(url).path),
                parents=parents,
                exclude_html=exclude_html
            )
        if exclude_html:
            return downloaded
    # no html file -> simply download this file
    with fname.open(mode="wb") as f:
        for chunk in req.iter_content(chunk_size=1024):
            f.write(chunk)
        return downloaded + [fname]

def download_ucimlr(identifier: str, outdir: Union[str | Path]="datasets", overwrite: bool=False) -> bool:
    outdir = Path(outdir).joinpath(identifier)
    if outdir.exists():
        return False
    url = f"https://archive.ics.uci.edu/ml/machine-learning-databases/{identifier}/"
    downloaded = download_recursive(url, outdir)
    for outfile in downloaded:
        if outfile.suffix in ['.Z', '.gz', '.zip', '.Z', '.tar', '.bz2']:
            # extract zip files
            extract_zip(outfile, outdir)

def load_csv_data(datadir: Path):
    namefiles = [x for x in outdir.iterdir() if ".names" in x.suffixes]
    # exclude .data.html files
    datafiles = [x for x in outdir.iterdir() if ".data" in x.suffixes and x.name.startswith(variant) and not x.suffix == ".html"]
    trainfiles = [x for x in outdir.iterdir() if ".train" in x.suffixes and x.name.startswith(variant)]
    if len(datafiles) == 0 and len(trainfiles) == 0:
        raise IOError(f"Could not find data or train file for UCIMLR database {identifier}, I do not know how to load this dataset. :(")
    elif len(datafiles) > 1:
        warnings.warn(f"Found multiple datafiles for UCIMLR database {identifier}: {datafiles}")
    elif len(trainfiles) > 1:
        warnings.warn(f"Found multiple training files for UCIMLR database {identifier}: {trainfiles}")
    columns = None if len(namefiles) == 0 else guess_ucimlr_columns(namefiles[0])
    if len(datafiles) > 0:
        return pd.read_csv(datafiles[0], names=columns, on_bad_lines="warn", encoding_errors="backslashreplace")
    else:
        # load training and test file
        trainfile = trainfiles[0]
        testfile = trainfile.parent / (trainfile.stem + ".test")
        if not testfile.exists():
            raise IOError(f"Could not find corresponding test file {testfile} for training gile {trainfile}")
        return [pd.read_csv(x, names=columns, on_bad_lines="warn", encoding_errors="backslashreplace") for x in [trainfile, testfile]]

def guess_ucimlr_columns(namefile):
    if not namefile.exists():
        return None
    try:
        text = namefile.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = namefile.read_text(encoding="latin-1")
    return None

import csv
csv.Sniffer()

def download_file(url, path):
    with requests.get(url, stream=True) as r:
        # raise error if HTTP error code was returned
        r.raise_for_status()
        with path.open(mode="wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)
