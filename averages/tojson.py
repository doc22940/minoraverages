import sys
import json
import decimal
import pathlib
from collections import OrderedDict

import numpy as np
import pandas as pd

def dropnull(rec):
    return {k: v
            for k, v, in rec.items()
            if isinstance(v, list) or not pd.isnull(v)}

def process_table(data):
    data.insert(0, 'row', np.arange(len(data))+1)
    data = data.drop(['year', 'nameLeague'], axis='columns') \
               .rename(columns={'nameClub': 'nameClub1'})
    for (i, col) in enumerate(data.columns):
        if col in ['AVG', 'SLG', 'PCT']:
            data[col] = data[col].apply(lambda x: ("%.3f" % x).lstrip("0"))
    return [dropnull(x) for x in data.to_dict(orient='records')]

name_map = {'Batting':      'player_batting',
            'Pitching':     'player_pitching',
            'Fielding':     'player_fielding',
            'TeamBatting':  'team_batting',
            'TeamPitching': 'team_pitching',
            'TeamFielding': 'team_fielding',
            'Standings':    'team_standings',
            'HeadToHead':   'team_head_to_head'}

class Int64Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            if not pd.isnull(obj):
                return int(obj)
            else:
                return None
        return json.JSONEncoder.default(self, obj)

def convert_workbook(fn):
    leagues = []
    tables = OrderedDict()
    for (name, data) in pd.read_excel(fn, sheet_name=None).items():
        if name == "Metadata": continue
        leagues.append(data[['year', 'nameLeague']].drop_duplicates())
        tables[name_map[name]] = process_table(data)

    league = pd.concat(leagues, ignore_index=True) \
               .drop_duplicates() \
               .to_dict(orient='records')[0]
    js = OrderedDict()
    js['league'] = league
    js['tables'] = tables
    return js

def main():
    source = pathlib.Path("transcript")/sys.argv[1]
    dest = pathlib.Path("json")/sys.argv[1]
    for fn in sorted(source.glob("*.xls")):
        print(fn)
        data = convert_workbook(fn)
        dest.mkdir(exist_ok=True, parents=True)
        with open(dest/(fn.name.replace(".xls", ".json")), "w") as f:
            f.write(json.dumps(data, indent=2, cls=Int64Encoder))
