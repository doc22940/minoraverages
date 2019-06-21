import sys
import json
from collections import OrderedDict

import pandas as pd

def dropnull(rec):
    return {k: v
            for k, v, in rec.items()
            if isinstance(v, list) or not pd.isnull(v)}

def process_table(data):
    data = [dropnull(x) for x in data.to_dict(orient='records')]
    return data

name_map = {'Batting':      'player_batting',
            'Pitching':     'player_pitching',
            'Fielding':     'player_fielding',
            'TeamBatting':  'team_batting',
            'TeamPitching': 'team_pitching',
            'TeamFielding': 'team_fielding',
            'Standings':    'team_standings'}
def main():
    js = OrderedDict()
    for (name, data) in pd.read_excel(sys.argv[1], sheet_name=None).items():
        if name == 'entries':
            continue
        js[name_map[name]] = process_table(data)
    
    js = json.dumps(js, indent=2)
    print(js)

