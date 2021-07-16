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
            )
            for n in names
        ]
        # get id where match was successful
        ids = [x.group(1) for x in ids if x is not None]
        for i in ids:
            print(i)
        data = cd.load_dataset("abalone", source="ucimlr")
        print(data)
