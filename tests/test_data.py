import unittest
import codaco.data as cd
import requests
import re
import pathlib

class TestUCImlr(unittest.TestCase):
    def test_guess_ucimlr_columns(self):
        outdir = pathlib.Path("~/temp/codaco").expanduser()
        outdir.mkdir(exist_ok=True, parents=True)
        # only requires download of a few files with no subfolders
        data = cd.load_dataset("abalone", source="ucimlr", download_to=outdir, force_download=True)
        # requires recursive download
        data = cd.load_dataset("mechanical-analysis", source="ucimlr", download_to=outdir, force_download=True)
        # requires recursive extraction
        data = cd.load_dataset("UNIX_user_data-mld", source="ucimlr", download_to=outdir, force_download=True)
