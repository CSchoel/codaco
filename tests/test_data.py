import unittest
import codaco.data as cd

class TestUCImlr(unittest.TestCase):
    def test_guess_ucimlr_columns(self):
        data = cd.load_dataset("abalone", source="ucimlr")
        print(data)
