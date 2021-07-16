import unittest
import codaco.data as cd
import requests
import re
import pathlib

class TestUCImlr(unittest.TestCase):
    def test_guess_ucimlr_columns(self):
        datasets = requests.get("https://archive.ics.uci.edu/ml/datasets.php").text
        names = re.findall(r'<a href="datasets\/(.*)"><img ', datasets)
        idfile = pathlib.Path("datasets/ucimlr.ids")
        if idfile.exists():
            ids = idfile.read_text(encoding="utf-8")
        else:
            ids = [
                re.search(
                    r'"..\/machine-learning-databases\/(.*[^/])\/?"',
                    requests.get("https://archive.ics.uci.edu/ml/datasets/" + n).text
                )
                for n in names
            ]
            # get id where match was successful
            ids = [x.group(1) for x in ids if x is not None]
            idfile.write_text("\n".join(ids), encoding="utf-8")
        for i in ids:
            print(i)
            data = cd.load_dataset(i, source="ucimlr")
