import os 
import json
from pathlib import Path

print(Path(__file__).parent / 'data')

def load_json(path):
    with open(path) as fp:
        return json.load(fp)

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

    def __init__(self, path=None):
        if path is None:
            self.path = Path(__file__).parent / 'data' / 'exp0'
        else:
            self.path = Path(path)

    def find_datafile(self, group, id):
        return next(self.path.glob(f"{group}/{id}.json"))

    def find_datafiles(self, group='*'):
        assert group in self.groups
        return list(self.path.glob(f"{group}/*.json"))

    @property
    def groups(self):
        return [p.parts[-1] for p in self.path.glob("*")]
    
    @property
    def recordings(self):
        return {g : self.find_datafiles(g) for g in self.groups}

    def __iter__(self):
        return iter(self.groups)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            group_index, animal_index = idx
            
            # get the group name
            if isinstance(group_index, int):
                group = self.groups[group_index]
            elif isinstance(group_index, str):
                group = group_index
                assert group in self.groups, Exception(f"{group} is not in {self.groups}!")
            else:
                raise Exception("Unrecognized group type!")

            # get the animal id
            if isinstance(animal_index, int):
                return load_json(self.recordings[group][animal_index])
            elif isinstance(animal_index, str):
                return load_json(self.find_datafile(group, animal_index))
            else:
                raise Exception("Unrecognized animal type!")
        elif isinstance(idx, int):
            return [r.parts[-1].split('.')[0] for r in self.recordings[self.groups[idx]]]
        elif isinstance(idx, str):
            return [r.parts[-1].split('.')[0] for r in self.recordings[idx]]

        