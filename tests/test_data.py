import unittest
import codaco.data as cd
import requests
import re
from pathlib import Path
import tempfile
import shutil

class TestUCImlr(unittest.TestCase):
    def setUp(self):
        self.data_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.data_dir)

    def test_download_recursive(self):
        """
        Test hypothesis: download_recursive fails to download content in subfolders
        """
        data = cd.load_dataset("mechanical-analysis", source="ucimlr", download_to=self.data_dir, force_download=True)
        expected =  {
            Path(self.data_dir) / 'mechanical-analysis/older-version/mechanical-analysis.notused-instances',
            Path(self.data_dir) / 'mechanical-analysis/older-version/mechanical-analysis.names',
            Path(self.data_dir) / 'mechanical-analysis/older-version/mechanical-analysis.data',
            Path(self.data_dir) / 'mechanical-analysis/older-version/Index',
            Path(self.data_dir) / 'mechanical-analysis/PUMPS-DATA-SET/DISTRIBUTION',
            Path(self.data_dir) / 'mechanical-analysis/PUMPS-DATA-SET/Index',
            Path(self.data_dir) / 'mechanical-analysis/Index'
        }
        self.assertEqual(expected, set(cd.walk(Path(self.data_dir) / "mechanical-analysis")))

    def test_extract_recursive(self):
        """
        Test hypothesis: extract_recursive fails to extract archives that are contained within another archive
        """
        data = cd.load_dataset("UNIX_user_data-mld", source="ucimlr", download_to=self.data_dir, force_download=True)
        expected = {
            Path(self.data_dir) / 'UNIX_user_data-mld/README',
            Path(self.data_dir) / 'UNIX_user_data-mld/UNIX_user_data.html',
            Path(self.data_dir) / 'UNIX_user_data-mld/UNIX_user_data/README',
            Path(self.data_dir) / 'UNIX_user_data-mld/UNIX_user_data/USER0',
            Path(self.data_dir) / 'UNIX_user_data-mld/UNIX_user_data/USER1',
            Path(self.data_dir) / 'UNIX_user_data-mld/UNIX_user_data/USER2',
            Path(self.data_dir) / 'UNIX_user_data-mld/UNIX_user_data/USER3',
            Path(self.data_dir) / 'UNIX_user_data-mld/UNIX_user_data/USER4',
            Path(self.data_dir) / 'UNIX_user_data-mld/UNIX_user_data/USER5',
            Path(self.data_dir) / 'UNIX_user_data-mld/UNIX_user_data/USER6',
            Path(self.data_dir) / 'UNIX_user_data-mld/UNIX_user_data/USER7',
            Path(self.data_dir) / 'UNIX_user_data-mld/UNIX_user_data/USER8'
        }
        self.assertEqual(expected, set(cd.walk(Path(self.data_dir) / "UNIX_user_data-mld")))
