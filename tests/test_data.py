import unittest
import codaco.data as cd
import requests
import re

class TestUCImlr(unittest.TestCase):
    def test_guess_ucimlr_columns(self):
        datasets = requests.get("https://archive.ics.uci.edu/ml/datasets.php").text
        names = re.findall(r'<a href="datasets\/(.*)"><img ', datasets)
        ids = [
            re.search(
                r'"..\/machine-learning-databases\/(.*[^/])\/?"',
                requests.get("https://archive.ics.uci.edu/ml/datasets/" + n).text
            ).group(1)
            for n in names
        ]
        for i in ids:
            print(n)
        data = cd.load_dataset("abalone", source="ucimlr")
        print(data)
