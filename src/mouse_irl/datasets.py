import os 
import json
from pathlib import Path

print(Path(__file__).parent / 'data')

class Dataset0:
    """Frances Sherratt's experimental data from Abizaid Lab @ Carleton
    
    Treatment Groups:
    - saline-saline
    - saline-ghrelin
    - mt2-ghrelin
    - mt2-saline

    Notes:
    - saline-ghrelin is the most active group. 
    - mt2-saline is the least active group.
    """

    def __init__(self):
        self.path = Path(__file__).parent / 'data' / 'exp0'
        
    def find_datafile(self, group, id):
        return next(self.path.glob(f"{group}/{id}.json"))

    def find_datafiles(self, group='*'):
        assert group in self.groups
        return list(self.path.glob(f"{group}/*.json"))

    @staticmethod
    def load(path):
        with open(path) as fp:
            return json.load(fp)

    @property
    def groups(self):
        return list(self.path.glob("*"))

