import sys
import json
from collections import OrderedDict

import numpy as np
import pandas as pd

def dropnull(rec):
    return {k: v
            for k, v, in rec.items()
            if isinstance(v, list) or not pd.isnull(v)}

def xform_league(data):
    return data.drop(['year', 'nameLeague'], axis='columns')

def xform_person(data):
    """Transform structure of person data columns.
    """
    if 'nameFirst' in data:
        data['nameLast'] = data.apply(lambda x:
                                      {'name':
                                           dropnull({'last': x['nameLast'],
                                                     'first': x['nameFirst']})},
                                      axis=1)
                                                
        del data['nameFirst']
    else:
        data['nameLast'] = data.apply(lambda x:
                                      {'name':
                                           dropnull({'last': x['nameLast']})},
                                      axis=1)
    return data.rename(columns={'nameLast': 'person'})

def xform_team_splits(data):
    data['splits'] = data.apply(lambda x:
                                [{"team": {"name": x[col]}}
                                 for col in data.columns
                                 if col.startswith('nameClub') and
                                    not pd.isnull(x[col]) and x[col] != "all"],
                                axis=1)
    return data[[c for c in data.columns if not c.startswith('nameClub')]]

def xform_totals(data):
    data['totals'] = [dropnull(x)
                      for x in data[[c for c in data.columns
                                    if c not in ['row', 'person', 'splits']]]
                                   .to_dict(orient='records')]
    return data[['row', 'person', 'totals', 'splits']]

def process_table(data):
    data = data.assign(row=np.arange(len(data))+1) \
               .pipe(xform_league) \
               .pipe(xform_person) \
               .pipe(xform_team_splits) \
               .pipe(xform_totals)
    data = [dropnull(x) for x in data.to_dict(orient='records')]
    return data

name_map = {'Batting':      'player_batting',
            'Pitching':     'player_pitching',
            'Fielding':     'player_fielding'}

def main():
    leagues = []
    tables = OrderedDict()
    for (name, data) in pd.read_excel(sys.argv[1], sheet_name=None).items():
        if name not in name_map: continue
        leagues.append(data[['year', 'nameLeague']].drop_duplicates())
        tables[name_map[name]] = process_table(data)

    league = pd.concat(leagues, ignore_index=True) \
               .drop_duplicates() \
               .to_dict(orient='records')[0]
    js = OrderedDict()
    js['league'] = league
    js['tables'] = tables
    js = json.dumps(js, indent=2)
    print(js)

