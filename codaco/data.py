from os import replace
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
import re
import io
from .fwf import replace_inline_tabs, guess_tabwidth, simple_table_blocks, get_table

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
        data = load_csv_data(Path(download_to) / "abalone")
        return data
    else:
        raise IOError("Unknown dataset source: {}".format(source))


def load_file(fname):
    """
    Loads a dataset from a file, detecting the file type based on the extension.
    """

    if fname.suffix == "csv":
        return pd.read_csv(fname)
    else:
        raise Exception("Sorry, I cannot load .{} files.".format(fname.suffix))


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

def download_ucimlr(identifier: str, outdir: Union[str | Path]="datasets", overwrite: bool=False):
    outdir = Path(outdir).joinpath(identifier)
    if outdir.exists() and not overwrite:
        return False
    url = f"https://archive.ics.uci.edu/ml/machine-learning-databases/{identifier}/"
    downloaded = download_recursive(url, outdir, overwrite=overwrite)
    for outfile in downloaded:
        if outfile.suffix in ['.Z', '.gz', '.zip', '.Z', '.tar', '.bz2']:
            # extract zip files
            extract_recursive(outfile, outdir)

def read_namefile(f: Path):
    # TODO handle non-utf8 files (maybe with chardet?)
    text = f.read_text(encoding="utf-8")
    if text.count("\t") > 0:
        # if file contains tabs, test with different tab sizes
        text = replace_inline_tabs(text, tabsize=guess_tabwidth(text))
    blocks = simple_table_blocks(text)
    return [get_table(text, s, e) for _,s,e in blocks]

def guess_column_names(f: Path, nattrib: int=None) -> List[str]:
    tables = read_namefile(f)
    for t in tables:
        if (nattrib is None or t.shape[1] == nattrib) and len([x for x in t.columns if not x.startswith("Unnamed")]) == t.shape[1]:
            # index in rows
            return t.columns
        elif (nattrib is None or t.shape[0] == nattrib) and len(set(t.iloc[:,0].values)) == t.shape[0]:
            # index in columns
            return t.iloc[:,0].values
    raise Exception(f"Could not find a table with row or column size {nattrib} in {f}.")

def load_csv_data(datadir: Path) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Finds all CSV formatted filed in datadir and loads them using pandas.
    """
    textfiles = [f for f in walk(datadir) if magic.from_file(f, mime=True) == 'text/plain']
    data = {}
    for f in filter(lambda x: magic.from_file(x, mime=True) in ["text/csv", "application/csv"], walk(datadir)):
        # TODO handle non-utf8 files (maybe with chardet?)
        sniffer = csv.Sniffer()
        head = sniffer.has_header(f.read_text("utf-8"))
        # NOTE for very large files we might want to replace this with sniffer.sniff()
        df = pd.read_csv(f, on_bad_lines="warn")
        nattrib = len(df.columns)
        if not head:
            # find most likely name file by naive position-based name matching
            namefile = max(textfiles, key=lambda x: sum([ a == b for a,b in zip(str(x), str(f))]))
            columns = guess_column_names(namefile, nattrib=nattrib)
            df = pd.read_csv(f, names=columns, on_bad_lines="warn")
        data[str(f.relative_to(datadir))] = df
    if len(data.keys()) == 1:
        return list(data.values())[0]
    else:
        return data

def download_file(url, path):
    with requests.get(url, stream=True) as r:
        # raise error if HTTP error code was returned
        r.raise_for_status()
        with path.open(mode="wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)
