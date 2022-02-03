import pandas as pd
import numpy as np
from pathlib import Path
import requests
from urllib.parse import urlparse, urljoin
import re
import warnings
import shutil
import subprocess
from typing import *
import magic
import csv

# TODO: function for loading ML-datasets as generators

def load_dataset(identifier, source="file", download_to="datasets", force_download: bool=False):
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
        download_ucimlr(identifier, outdir=download_to, overwrite=force_download)
        return [] # TODO implement data loading
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


def walk(path: Path) -> List[Path]:
    for p in path.iterdir():
        if p.is_file():
            yield p
        else:
            yield from walk(p)

def extract_recursive(filepath: Path, outdir: Path):
    known = list(walk(outdir))
    print(f"extracting {filepath}")
    # extract zip files
    try:
        shutil.unpack_archive(filepath, outdir)
        filepath.unlink()
    except shutil.ReadError as e:
        ret = subprocess.run(["uncompress", "-f", str(filepath.absolute())], cwd=outdir, capture_output=True)
        if ret.returncode != 0:
            raise IOError(f"could not extract {filepath.absolute()}")
        # check if extracted files need to be extracted again
    for f in (x for x in walk(outdir) if x not in known and x.suffix in [".zip", ".gz", ".tar", ".bz2", ".Z"]):
        extract_recursive(f, outdir)

def download_recursive(
        url: str, outdir: Path, parents: bool=False, exclude_html: bool=True,
        overwrite: bool=False, base_path: Path=None, leave_site=False,
        visited: Set[str]=None) -> List[Path]:
    parsed = urlparse(url)
    path = Path(parsed.path)
    if visited is None:
        visited = set()
    if url in visited:
        return []
    visited.add(url)
    if base_path is None:
        # NOTE: this is sensitive to slashes at the end of the URL
        base_path = path
    if parents:
        # if we may traverse up until the root, we need to have subdirectories
        fname = outdir / parsed.netloc / parsed.path
    else:
        # otherwise we may cut away the base path and do not need the site
        fname = outdir / path.relative_to(base_path)
    fname.parent.mkdir(exist_ok=True, parents=True)
    if fname.exists() and not overwrite:
        return []
    downloaded = []
    req = requests.get(url)
    # content type text/html indicates that we need to look for links
    if req.headers['content-type'].startswith("text/html"):
        # find all hrefs to trigger recursive download
        text = req.text
        refs = re.findall(r"href\s*=\s*\"(.+?)\"", text)
        for r in refs:
            refurl = urljoin(url, r)
            refpath = Path(urlparse(refurl).path)
            # avoid parents
            if url.startswith(refurl) and not parents:
                continue
            # avoid leaving main site
            if urlparse(refurl).netloc != parsed.netloc and not leave_site:
                continue
            downloaded += download_recursive(
                refurl,
                outdir,
                parents=parents,
                exclude_html=exclude_html,
                overwrite=overwrite,
                base_path=base_path,
                leave_site=leave_site,
                visited=visited
            )
        if exclude_html:
            return downloaded
    # no html file or HTML file is requested -> simply download this file
    print(f"downloading {url}")
    with fname.open(mode="wb") as f:
        for chunk in req.iter_content(chunk_size=1024):
            f.write(chunk)
        return downloaded + [fname]

def download_ucimlr(identifier: str, outdir: Union[str | Path]="datasets", overwrite: bool=False) -> bool:
    outdir = Path(outdir).joinpath(identifier)
    if outdir.exists() and not overwrite:
        return False
    url = f"https://archive.ics.uci.edu/ml/machine-learning-databases/{identifier}/"
    downloaded = download_recursive(url, outdir, overwrite=overwrite)
    for outfile in downloaded:
        if outfile.suffix in ['.Z', '.gz', '.zip', '.Z', '.tar', '.bz2']:
            # extract zip files
            extract_recursive(outfile, outdir)

def read_namefile(f: Path, nattrib: Union[int, None]=None):
    if nattrib is None:
        raise Exception("Currently reading namefiles without knowing the number of attributes to look for is not implemented")
    # TODO handle non-utf8 files (maybe with chardet?)
    text = f.read_text(encoding="utf-8")
    if text.count("\t") > 0:
        # if file contains tabs, test with different tab sizes
        success = find_table_block(text, tabsize=2)
        if not success:
            success = find_table_block(text, tabsize=4)
        if not success:
            success = find_table_block(text, tabsize=8)
    else:
        success = find_table_block(text)
    return success

def find_table_block(text: str, tabsize: int=4):
    # replace tabs by spaces
    text = text.replace("\t", " " * tabsize)
    # find longest consecutive number of lines where more than one column consists entirely of spaces
    lastline = {}
    found = []
    def select_columns(continuation: Dict[int, int]):
        # only keep columns of at least height 3
        values = filter(lambda x: x[1] > 2, continuation.items())
        # only keep local maxima
        # NOTE: plateaus are discarded by taking leftmost value
        values = filter(lambda x: x[1] > continuation.get(x[0]-1, 0) and x[1] >= continuation.get(x[0]+1, 0), values)
        return dict(values)
    def max_cells(continuation: Dict[int, int]):
        # successively test how many cells a table of height v
        # would have for each column height v in continuation
        options = []
        cols_by_height = {}
        for k, v in continuation.items():
            cols_by_height.setdefault(v, [])
            cols_by_height[v].append(k)
        # by staring with the highest column, we know that we
        # only have to add more column indices, not substract those
        # that are not high enough
        cbh_sorted = sorted(cols_by_height.items(), reverse=True)
        indices = []
        for v, ks in cbh_sorted:
            indices += ks
            n = v * (len(indices) + 1)
            options.append((n, indices[:], v))
        return max(options)
    for i, l in enumerate(text.splitlines()):
        colcount = {j: lastline.get(j, 0) + 1 for j, c in enumerate(l) if c == " "}
        # remove leading indices, because those can stem from indentation
        j = 0
        while j in colcount:
            del colcount[j]
            j += 1
        # check if we are at the end of a consecutive run
        # => i.e. the maximum runlenght of the previous line was higher
        if max(lastline.values(), default=0) > max(colcount.values(), default=0):
            # no continuing lines found
            found.append((i-1, sum(x for x in lastline.values() if x == max(lastline.values()))))
        lastline = colcount
    print(found)
    best = max(found, key=lambda x: x[1], default=(0, 0))
    if best[1] < 20:
        return False
    return best

def load_csv_data(datadir: Path):
    """
    Finds all CSV formatted filed in datadir and loads them using pandas.
    """
    for f in filter(lambda x: magic.from_file(x) == "CSV text", walk(datadir)):
        sniffer = csv.Sniffer()
        # TODO handle non-utf8 files (maybe with chardet?)
        head = sniffer.has_header(f.read_text("utf-8"))
        print(f, head)
    return

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
