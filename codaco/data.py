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
        ret = subprocess.run(["uncompress", "-f", str(filepath.absolute())], cwd=outdir, capture_output=True)
        if ret.returncode != 0:
            raise IOError(f"could not extract {filepath.absolute()}")
        # check if extracted files need to be extracted again
    for f in (x for x in outdir.iterdir() if x not in known and x.suffix in [".zip", ".gz", ".tar", ".bz2", ".Z"]):
        extract_zip(f, outdir)


def load_ucimlr(identifier, download_to="datasets", variant="", force=False):
    exclude = [
        "artificial-characters",
        "audiology",
        "chess/king-rook-vs-king-knight",
        "chess/domain-theories",
        "chorales",
        "diabetes",
        "dgp-2",
        "document-understanding",
        "ebl",
        "heart-disease",
        "function-finding",
        "icu",
        "image", # needs advanced CSV handling
        "internet_ads", # should work with encoding_errors = 'backslashreplace'
        "led-display-creator", # c code
        "logic-theorist", # includes code, requires recursive download
        "mechanical-analysis", # requires recursive download
        "mobile-robots", # not a tabular format
        "molecular-biology/protein-secondary-structure", # sequential data
        "mfeat", # data file names have no suffix
        "othello", # some logical code instead of data
        "optdigits", # .tes/.tra instead of .test/.train
        "pendigits", # .tes/.tra instead of .test/.train
        "qsar", # tar file with no suffix
        "quadrapeds", # c code
        "solar-flare", # .data1/.data2 instead of .data
        "statlog", # requires recursive download
        "student-loan", # perl code
        "undocumented", # requires recursive download
        "auslan-mld", # data in many subfolders
        "auslan2-mld", # data in many subfolders
        "census1990-mld", # works, but is quite large (360 MB text files)
        "CorelFeatures-mld", # .asc instead of .data
        "ecoli-mld", # perl code
        "eeg-mld", # data in many subfolders
        "faces-mld", # image data in many subfolders
        "tic-mld", # custom name for data files
        "entree-mld", # data in many subfolders
        "el_nino-mld", # .dat instead of .data
        "internet_usage-mld", # .dat instead of .data
        "20newsgroups-mld", # text corpus in subfolders
        "ipums-mld", # custom name for data files
        "kddcup98-mld", # requires recursive download
        "tb-mld", # perl code
        "movies-mld", # requires recursive download
        "msnbc-mld", # special data format
        "nsfabs-mld", # text data in many subfolders
        "reuters21578-mld", # text data
        "SyskillWebert-mld", # text data in many subfolders
        "UNIX_user_data-mld", # text data in subfolders
        "volcanoes-mld", # complex data format in many subfolders
        "statlog/australian", # .dat instead of .data
        "statlog/heart", # .dat instead of .data
        "statlog/satimage", # .trn/tst instead of .train/test
        "statlog/segment", # .dat instead of .data
        "statlog/shuttle", # .trn/tst instead of .train/test
        "statlog/vehicle", # .dat instead of .data
        "undocumented/connectionist-bench/sonar", # -data instead of .data
        "undocumented/pazzani", # just a single file with lisp code
        "undocumented/taylor", # requires advanced CSV handling
        "undocumented/sigillito", # sequential data
        "uji-penchars/version1", # subfolders
        "forest-fires", # .csv instead of .data
    ]
    if identifier in exclude:
        return None
    outdir = pathlib.Path(download_to).joinpath(identifier)
    if not outdir.exists() or force:
        url = f"https://archive.ics.uci.edu/ml/machine-learning-databases/{identifier}/"
        refs = re.findall(r"href=\"(.+?)\"", requests.get(url).text)
        refs = [x for x in refs if not x.startswith('/') and x not in ["Index"]]
        print(refs)
        outdir.mkdir(parents=True, exist_ok=True)
        for f in refs:
            if f == "Index":
                continue
            outfile = outdir.joinpath(f)
            download_file(url + f, outfile)
            if outfile.suffix in ['.Z', '.gz']:
                # extract zip files
                extract_zip(outfile, outdir)

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
    text = namefile.read_text(encoding="utf-8")
    return None

def download_file(url, path):
    with requests.get(url, stream=True) as r:
        # raise error if HTTP error code was returned
        r.raise_for_status()
        with path.open(mode="wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)
