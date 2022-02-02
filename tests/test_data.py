import unittest
import codaco.data as cd
import requests
import re
import pathlib

class TestUCImlr(unittest.TestCase):
    def test_guess_ucimlr_columns(self):
        outdir = pathlib.Path("~/temp/codaco").expanduser()
        outdir.mkdir(exist_ok=True, parents=True)
        data = cd.load_dataset("abalone", source="ucimlr", download_to=outdir, force_download=True)
        data = cd.load_dataset("chess", source="ucimlr", download_to=outdir, force_download=True)
