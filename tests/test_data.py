import unittest
import codaco.data as cd
import requests
import re
import pathlib

class TestUCImlr(unittest.TestCase):
    def test_guess_ucimlr_columns(self):
        outdir = pathlib.Path("~/temp/codaco").expanduser()
        outdir.mkdir(exist_ok=True, parents=True)
        datasets = requests.get("https://archive.ics.uci.edu/ml/datasets.php").text
        names = re.findall(r'<a href="datasets\/(.*)"><img ', datasets)
        idfile = outdir.joinpath("ucimlr.ids")
        if idfile.exists():
            ids = idfile.read_text(encoding="utf-8").splitlines()
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
        exceptional = []
        for i in ids:
            print(i)
            data = cd.load_dataset(i, source="ucimlr", download_to=outdir)
            try:
                data = cd.load_dataset(i, source="ucimlr", download_to=outdir)
            except Exception as e:
                print(f"ERROR on dataset {i}: {e}")
                exceptional.append(i)
        print(f"Exceptional IDs: {exceptional}")
