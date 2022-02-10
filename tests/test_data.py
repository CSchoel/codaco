import unittest
import codaco.data as cd
import requests
import re
from pathlib import Path
import tempfile
import shutil
import codaco.stat as cs
import pandas as pd

class TestUCImlr(unittest.TestCase):
    def setUp(self):
        self.data_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.data_dir)

    def test_download_recursive(self):
        """
        Test hypothesis: download_recursive fails to download content in subfolders
        """
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mechanical-analysis/"
        dlpath = self.data_dir / "mechanical-analysis"
        downloaded = cd.download_recursive(url, outdir=dlpath, overwrite=True)
        expected =  {
            self.data_dir / 'mechanical-analysis/older-version/mechanical-analysis.notused-instances',
            self.data_dir / 'mechanical-analysis/older-version/mechanical-analysis.names',
            self.data_dir / 'mechanical-analysis/older-version/mechanical-analysis.data',
            self.data_dir / 'mechanical-analysis/older-version/Index',
            self.data_dir / 'mechanical-analysis/PUMPS-DATA-SET/DISTRIBUTION.Z',
            self.data_dir / 'mechanical-analysis/PUMPS-DATA-SET/Index',
            self.data_dir / 'mechanical-analysis/Index'
        }
        self.assertEqual(expected, set(downloaded))

    def test_extract_recursive(self):
        """
        Test hypothesis: extract_recursive fails to extract archives that are contained within another archive
        """
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/UNIX_user_data-mld/"
        dlpath = self.data_dir / "UNIX_user_data-mld"
        downloaded = cd.download_recursive(url, outdir=dlpath, overwrite=True)
        cd.extract_recursive(dlpath / "UNIX_user_data.tar.gz", outdir=dlpath)
        expected = {
            self.data_dir / 'UNIX_user_data-mld/README',
            self.data_dir / 'UNIX_user_data-mld/UNIX_user_data.html',
            self.data_dir / 'UNIX_user_data-mld/UNIX_user_data/README',
            self.data_dir / 'UNIX_user_data-mld/UNIX_user_data/USER0',
            self.data_dir / 'UNIX_user_data-mld/UNIX_user_data/USER1',
            self.data_dir / 'UNIX_user_data-mld/UNIX_user_data/USER2',
            self.data_dir / 'UNIX_user_data-mld/UNIX_user_data/USER3',
            self.data_dir / 'UNIX_user_data-mld/UNIX_user_data/USER4',
            self.data_dir / 'UNIX_user_data-mld/UNIX_user_data/USER5',
            self.data_dir / 'UNIX_user_data-mld/UNIX_user_data/USER6',
            self.data_dir / 'UNIX_user_data-mld/UNIX_user_data/USER7',
            self.data_dir / 'UNIX_user_data-mld/UNIX_user_data/USER8'
        }
        self.assertEqual(expected, set(cd.walk(Path(self.data_dir) / "UNIX_user_data-mld")))

    def test_load_csv(self):
        """
        Test hypothesis: load_csv_data fails to identify csv files or falsely identifies plain text files as csv.
        """
        cd.download_ucimlr("abalone", self.data_dir)
        dat = cd.load_csv_data(self.data_dir / "abalone")
        for k in dat:
            print(k)
            print(dat[k])
        print(list(cd.walk(self.data_dir)))
        if not isinstance(dat, pd.DataFrame):
            self.fail(f"Expected DataFrame, but got {type(dat)}")
        cols = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
        self.assertEqual(cols, list(dat.columns))
        first = ['M', 0.455, 0.365, 0.095, 0.514, 0.2245, 0.101, 0.15, 15]
        last = ['M', 0.71, 0.555, 0.195, 1.9485, 0.9455, 0.3765, 0.495, 12]
        self.assertEqual(first, list(dat.iloc[0,:].values))
        self.assertEqual(last, list(dat.iloc[-1,:].values))

    def test_guess_column_names(self):
        """
        Test hypothesis: read_namefile fails to find attribute names in block formatted name file.
        """
        cd.download_ucimlr("abalone", self.data_dir)
        nf = cd.guess_column_names(Path(self.data_dir) / "abalone/abalone.names", nattrib=9)
        cols = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
        self.assertEqual(cols, list(nf))

    def test_load_dataset(self):
        """
        Test hypothesis: load_dataset is unable to load a simple UCIMLR dataset
        that must be downloaded before it can be used.
        """
        data = cd.load_dataset("abalone", source="ucimlr", download_to=self.data_dir)
        if not isinstance(data, pd.DataFrame):
            self.fail(f"Expected DataFrame, but got {type(data)}")
        cols = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
        self.assertEqual(cols, list(data.columns))
        self.assertEqual((4177, 9), data.shape)
