"""Process source transcriptions into standardised output formats.

Copyright (c) 2016, Dr T L Turocy (ted.turocy@gmail.com)
                    Chadwick Baseball Bureau (http://www.chadwick-bureau.com)

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
"""

import xlrd
import pandas as pd
import damm

class Workbook(object):
    def __init__(self, fn):
        self.fn = fn

    @staticmethod
    def _standardize_columns(dataframe, columns):
        for col in columns:
            if col not in dataframe:
                dataframe[col] = None
        return dataframe[columns]

    @staticmethod
    def _compute_stints(multiclub, g_label):
        clubs = pd.melt(multiclub[['person.ref'] +
                                  [x for x in multiclub.columns if x.startswith("nameClub")]],
                        id_vars='person.ref')
        clubs = clubs[~clubs['value'].isnull()]
        clubs[g_label] = clubs['value'].apply(lambda x: x.split("@")[0] if "@" in x else None)
        clubs['value'] = clubs['value'].str.split("@").str[-1]
        clubs['S_STINT'] = clubs['variable'].str[-1]
        clubs = clubs[['person.ref', 'value', 'S_STINT', g_label]]
        clubs.columns = ['person.ref', 'nameClub1', 'S_STINT', g_label]
        return clubs

    @property
    def individual_batting(self):
        with open(self.fn) as f:
            df = pd.read_excel(f, sheetname='Batting')
        df['person.ref'] = (~df['nameLast'].isnull()).cumsum(). \
                           apply(lambda x: 'B%04d%d' %
                                 (x, damm.encode("%04d" % x)))

        if 'nameClub2' in df:
            df['S_STINT'] = df['nameClub2'].apply(lambda x: 'T' if not pd.isnull(x) else '0')
        else:
            df['S_STINT'] = '0'
        multiclub = df[df['S_STINT'] == 'T']
        clubs = self._compute_stints(multiclub, 'G')
        df = pd.concat([df, clubs], ignore_index=True)
        df['year'] = df['year'].fillna(method='pad')
        df['nameLeague'] = df['nameLeague'].fillna(method='pad')
        df.sort_values(['person.ref', 'S_STINT'], inplace=True)
        for col in ['nameLast', 'nameFirst', 'phase.name', 'bats']:
            if col in df:
                df[col] = df.groupby('person.ref')[col].fillna(method='backfill')
        df.loc[df['S_STINT'] == 'T', 'nameClub1'] = None

        if 'Pos' in df:
            for pos in ['P', 'C', '1B', '2B', '3B', 'SS', 'OF']:
                df['F_%s_POS' % pos] = df[~df.Pos.isnull()]['Pos'].apply(lambda x: 1 if pos in x.split("-") else 0)
        # These are captured as YYYYMMDD - make sure they are treated as
        # strings and not floats
        for col in ['dateFirst', 'dateLast']:
            if col in df:
                df[col] = df[col].apply(lambda x: str(int(x)) if not pd.isnull(x) else x)

        df.rename(inplace=True,
                  columns={'year':         'league.year',
                           'nameLeague':   'league.name',
                           'nameClub1':    'entry.name',
                           'nameLast':     'person.name.last',
                           'nameFirst':    'person.name.given',
                           'bats':         'person.bats',
                           'dateFirst':    'S_FIRST',
                           'dateLast':     'S_LAST',
                           'G':            'B_G',
                           'AB':           'B_AB',
                           'R':            'B_R',
                           'ER':           'B_ER',
                           'H':            'B_H',
                           'TB':           'B_TB',
                           'H1B':          'B_1B',
                           'H2B':          'B_2B',
                           'H3B':          'B_3B',
                           'HR':           'B_HR',
                           'RBI':          'B_RBI',
                           'BB':           'B_BB',
                           'IBB':          'B_IBB',
                           'SO':           'B_SO',
                           'GDP':          'B_GDP',
                           'HP':           'B_HP',
                           'SH':           'B_SH',
                           'SF':           'B_SF',
                           'SB':           'B_SB',
                           'CS':           'B_CS',
                           'AVG':          'B_AVG',
                           'AVG_RANK':     'B_AVG_RANK'})
        return df

    @property
    def individual_pitching(self):
        with open(self.fn) as f:
            df = pd.read_excel(f, sheetname='Pitching')
        df['person.ref'] = (~df['nameLast'].isnull()).cumsum().apply(lambda x: 'P%04d%d' % (1000+x, damm.encode("%04d" % (1000+x))))

        if 'nameClub2' in df:
            df['S_STINT'] = df['nameClub2'].apply(lambda x: 'T' if not pd.isnull(x) else '0')
        else:
            df['S_STINT'] = '0'
        multiclub = df[df['S_STINT'] == 'T']
        clubs = self._compute_stints(multiclub, 'GP')
        df = pd.concat([df, clubs], ignore_index=True)
        df['year'] = df['year'].fillna(method='pad')
        df['nameLeague'] = df['nameLeague'].fillna(method='pad')
        df.sort_values(['person.ref', 'S_STINT'], inplace=True)
        for col in ['nameLast', 'nameFirst', 'phase.name', 'throws']:
            if col in df:
                df[col] = df.groupby('person.ref')[col].fillna(method='backfill')
        df.loc[df['S_STINT'] == 'T', 'nameClub1'] = None

        df.rename(inplace=True,
                  columns={'year':         'league.year',
                           'nameLeague':   'league.name',
                           'nameClub1':    'entry.name',
                           'nameLast':     'person.name.last',
                           'nameFirst':    'person.name.given',
                           'throws':       'person.throws',
                           'GP':           'P_G',
                           'GS':           'P_GS',
                           'CG':           'P_CG',
                           'SHO':          'P_SHO',
                           'GF':           'P_GF',
                           'TO':           'P_TO',
                           'W':            'P_W',
                           'L':            'P_L',
                           'T':            'P_T',
                           'PCT':          'P_PCT',
                           'IP':           'P_IP',
                           'AB':           'P_AB',
                           'H':            'P_H',
                           'R':            'P_R',
                           'ER':           'P_ER',
                           'HR':           'P_HR',
                           'BB':           'P_BB',
                           'IBB':          'P_IBB',
                           'SO':           'P_SO',
                           'HB':           'P_HP',
                           'SH':           'P_SH',
                           'SF':           'P_SF',
                           'WP':           'P_WP',
                           'ERA':          'P_ERA',
                           'BK':           'P_BK',
                           'SB':           'P_SB',
                           'ERA_RANK':     'P_ERA_RANK'})
        return df

    @property
    def individual_fielding(self):
        with open(self.fn) as f:
            try:
                df = pd.read_excel(f, sheetname='Fielding')
            except xlrd.biffh.XLRDError:
                return pd.DataFrame(columns=['league.year'])
        df['person.ref'] = (~df['nameLast'].isnull()).cumsum().apply(lambda x: 'F%04d%d' % (2000+x, damm.encode("%04d" % (2000+x))))

        if 'nameClub2' in df:
            df['S_STINT'] = df['nameClub2'].apply(lambda x: 'T' if not pd.isnull(x) else '0')
        else:
            df['S_STINT'] = '0'
        multiclub = df[df['S_STINT'] == 'T']
        clubs = self._compute_stints(multiclub, 'G')
        df = pd.concat([df, clubs], ignore_index=True)
        df['year'] = df['year'].fillna(method='pad')
        df['nameLeague'] = df['nameLeague'].fillna(method='pad')
        df.sort_values(['person.ref', 'S_STINT'], inplace=True)
        for col in ['nameLast', 'nameFirst', 'Pos', 'phase.name', 'throws']:
            if col in df:
                df[col] = df.groupby('person.ref')[col].fillna(method='backfill')
        df.loc[df['S_STINT'] == 'T', 'nameClub1'] = None

        df['POS'] = 1
        df['rowid'] = df['POS'].cumsum()
        melted = pd.melt(df, id_vars=['rowid', 'Pos'])
        # There are some circumstances in which leagues reported only
        # total fielding data but primary positions (the 1947 Coastal Plain
        # League is one example).  If an explicit F_ALL is used at the start
        # of the column, we will respect that.
        # The effect will therefore be that we can get a by-position POS
        # entry for the primary position, but record the aggregate stats.
        melted['variable'] = melted.apply(lambda x:
                                          ("F_%s_%s" %
                                           (x['Pos'], x['variable']))
                                          if not x['variable'].startswith("ALL")
                                          else ("F_%s" % x['variable']),
                                          axis=1)
        melted = melted.pivot(columns='variable', values='value', index='rowid')
        df = pd.merge(df, melted, left_on='rowid', right_index=True)
        df.rename(inplace=True,
                  columns={'year':         'league.year',
                           'nameLeague':   'league.name',
                           'nameClub1':    'entry.name',
                           'nameLast':     'person.name.last',
                           'nameFirst':    'person.name.given',
                           'throws':       'person.throws'})
        return df

    @property
    def individual_playing(self):
        df = pd.concat([self.individual_batting,
                        self.individual_pitching,
                        self.individual_fielding],
                       ignore_index=True)
        if 'phase.name' not in df:
            df['phase.name'] = 'regular'
        df['phase.name'] = df['phase.name'].fillna('regular')
        cols = ['league.year', 'league.name',
                'person.ref',
                'person.name.last', 'person.name.given',
                'person.bats', 'person.throws',
                'phase.name', 'S_STINT', 'entry.name',
                'S_FIRST', 'S_LAST',
                'B_G', 'B_AB', 'B_R', 'B_ER', 'B_H', 'B_TB',
                'B_1B', 'B_2B', 'B_3B', 'B_HR', 'B_RBI',
                'B_BB', 'B_IBB', 'B_SO', 'B_GDP', 'B_HP', 'B_SH', 'B_SF',
                'B_SB', 'B_CS',
                'B_AVG', 'B_AVG_RANK',
                'P_G', 'P_GS', 'P_CG', 'P_SHO', 'P_TO', 'P_GF',
                'P_W', 'P_L', 'P_T', 'P_PCT',
                'P_IP', 'P_TBF', 'P_AB', 'P_R', 'P_ER', 'P_H',
                'P_HR', 'P_BB', 'P_IBB', 'P_SO', 'P_HP', 'P_SH',
                'P_WP', 'P_BK', 'P_SB',
                'P_ERA', 'P_ERA_RANK',
                'F_1B_POS', 'F_1B_G', 'F_1B_TC', 'F_1B_PO', 'F_1B_A', 'F_1B_E',
                'F_1B_DP', 'F_1B_TP', 'F_1B_PCT',
                'F_2B_POS', 'F_2B_G', 'F_2B_TC', 'F_2B_PO', 'F_2B_A', 'F_2B_E',
                'F_2B_DP', 'F_2B_TP', 'F_2B_PCT',
                'F_3B_POS', 'F_3B_G', 'F_3B_TC', 'F_3B_PO', 'F_3B_A', 'F_3B_E',
                'F_3B_DP', 'F_3B_TP', 'F_3B_PCT',
                'F_SS_POS', 'F_SS_G', 'F_SS_TC', 'F_SS_PO', 'F_SS_A', 'F_SS_E',
                'F_SS_DP', 'F_SS_TP', 'F_SS_PCT',
                'F_OF_POS', 'F_OF_G', 'F_OF_TC', 'F_OF_PO', 'F_OF_A', 'F_OF_E',
                'F_OF_DP', 'F_OF_TP', 'F_OF_PCT',
                'F_LF_POS', 'F_LF_G', 'F_LF_TC', 'F_LF_PO', 'F_LF_A', 'F_LF_E',
                'F_LF_DP', 'F_LF_TP', 'F_LF_PCT',
                'F_CF_POS', 'F_CF_G', 'F_CF_TC', 'F_CF_PO', 'F_CF_A', 'F_CF_E',
                'F_CF_DP', 'F_CF_TP', 'F_CF_PCT',
                'F_RF_POS', 'F_RF_G', 'F_RF_TC', 'F_RF_PO', 'F_RF_A', 'F_RF_E',
                'F_RF_DP', 'F_RF_TP', 'F_RF_PCT',
                'F_C_POS', 'F_C_G', 'F_C_INN', 'F_C_TC', 'F_C_PO', 'F_C_A', 'F_C_E',
                'F_C_DP', 'F_C_TP', 'F_C_PB', 'F_C_SB', 'F_C_CS', 'F_C_PCT',
                'F_P_POS', 'F_P_G', 'F_P_TC', 'F_P_PO', 'F_P_A', 'F_P_E',
                'F_P_DP', 'F_P_TP', 'F_P_PCT',
                'F_ALL_G', 'F_ALL_TC', 'F_ALL_PO', 'F_ALL_A', 'F_ALL_E',
                'F_ALL_DP', 'F_ALL_TP', 'F_ALL_PCT']
        return self._standardize_columns(df, cols)

    _individual_managing_columns = \
      ['league.year', 'league.name',
       'entry.name', 'seq', 'person.ref',
       'person.name.last', 'person.name.given',
       'S_FIRST', 'S_LAST']

    @property
    def individual_managing(self):
        with open(self.fn) as f:
            try:
                df = pd.read_excel(f, sheetname='Managing')
            except xlrd.biffh.XLRDError:
                return pd.DataFrame(columns=self._individual_managing_columns)
        df['person.ref'] = (~df['nameLast'].isnull()).cumsum().apply(lambda x: 'M%04d%d' % (9000+x, damm.encode("%04d" % (9000+x))))
        df.rename(inplace=True,
                  columns={'year':          'league.year',
                           'nameLeague':    'league.name',
                           'nameClub':      'entry.name',
                           'nameLast':      'person.name.last',
                           'nameFirst':     'person.name.given',
                           'phase':         'phase.name',
                           'dateFirst':     'S_FIRST',
                           'dateLast':      'S_LAST'})
        if 'phase.name' not in df:
            df['phase.name'] = 'regular'
        return self._standardize_columns(df, self._individual_managing_columns)

    @property
    def _team_standings(self):
        with open(self.fn) as f:
            try:
                df = pd.read_excel(f, sheetname='Standings')
            except xlrd.biffh.XLRDError:
                return pd.DataFrame(columns=['league.year'])
        df.rename(inplace=True,
                  columns={'year':    'league.year',
                           'nameLeague':  'league.name',
                           'nameClub':  'entry.name',
                           'phase':     'phase.name',
                           'division':  'division.name',
                           'W':       'R_W',
                           'L':       'R_L',
                           'T':       'R_T',
                           'PCT':     'R_PCT',
                           'RANK':    'R_RANK'})
        if 'phase.name' not in df:
            df['phase.name'] = 'regular'
        return df

    @property
    def _team_batting(self):
        with open(self.fn) as f:
            try:
                df = pd.read_excel(f, sheetname='TeamBatting')
            except xlrd.biffh.XLRDError:
                return pd.DataFrame(columns=['league.year'])
        df.rename(inplace=True,
                  columns={'year':     'league.year',
                           'nameLeague':  'league.name',
                           'nameClub':    'entry.name',
                           'phase':    'phase.name',
                           'G':        'B_G',
                           'IP':       'B_IP',
                           'AB':       'B_AB',
                           'R':        'B_R',
                           'ER':       'B_ER',
                           'OR':       'P_R',
                           'H':        'B_H',
                           'TB':       'B_TB',
                           'H1B':      'B_1B',
                           'H2B':      'B_2B',
                           'H3B':      'B_3B',
                           'HR':       'B_HR',
                           'SH':       'B_SH',
                           'SB':       'B_SB',
                           'BB':       'B_BB',
                           'HP':       'B_HP',
                           'SO':       'B_SO',
                           'RBI':      'B_RBI',
                           'LOB':      'B_LOB',
                           'AVG':      'B_AVG'})
        if 'phase.name' not in df:
            df['phase.name'] = 'regular'
        return df

    @property
    def _team_pitching(self):
        with open(self.fn) as f:
            try:
                df = pd.read_excel(f, sheetname='TeamPitching')
            except xlrd.biffh.XLRDError:
                return pd.DataFrame(columns=['league.year'])
        df.rename(inplace=True,
                  columns={'year':     'league.year',
                           'nameLeague':  'league.name',
                           'nameClub':    'entry.name',
                           'phase':    'phase.name',
                           'GP':       'P_G',
                           'CG':       'P_CG',
                           'SHO':      'P_SHO',
                           'GF':       'P_GF',
                           'W':        'P_W',
                           'L':        'P_L',
                           'T':        'P_T',
                           'PCT':      'P_PCT',
                           'IP':       'P_IP',
                           'AB':       'P_AB',
                           'R':        'P_R',
                           'ER':       'P_ER',
                           'H':        'P_H',
                           'HR':       'P_HR',
                           'BB':       'P_BB',
                           'IBB':      'P_IBB',
                           'SO':       'P_SO',
                           'HB':       'P_HP',
                           'SH':       'P_SH',
                           'SF':       'P_SF',
                           'WP':       'P_WP',
                           'BK':       'P_BK',
                           'ERA':      'P_ERA'})
        if 'phase.name' not in df:
            df['phase.name'] = 'regular'
        return df

    @property
    def _team_fielding(self):
        with open(self.fn) as f:
            try:
                df = pd.read_excel(f, sheetname='TeamFielding')
            except xlrd.biffh.XLRDError:
                return pd.DataFrame(columns=['league.year'])
        df.rename(inplace=True,
                  columns={'year':     'league.year',
                           'nameLeague':  'league.name',
                           'nameClub':    'entry.name',
                           'phase':    'phase.name',
                           'G':        'F_G',
                           'TC':       'F_TC',
                           'PO':       'F_PO',
                           'A':        'F_A',
                           'E':        'F_E',
                           'DP':       'F_DP',
                           'TP':       'F_TP',
                           'PB':       'F_PB',
                           'CI':       'F_XI',
                           'LOB':      'F_LOB',
                           'PCT':      'F_PCT'})
        if 'phase.name' not in df:
            df['phase.name'] = 'regular'
        return df

    @property
    def _team_attendance(self):
        with open(self.fn) as f:
            try:
                df = pd.read_excel(f, sheetname='Attendance')
            except xlrd.biffh.XLRDError:
                return pd.DataFrame(columns=['league.year'])
        df.rename(inplace=True,
                  columns={'year':        'league.year',
                           'nameLeague':  'league.name',
                           'nameClub':    'entry.name',
                           'phase':       'phase.name',
                           'ATT':         'R_ATT'})
        if 'phase.name' not in df:
            df['phase.name'] = 'regular'
        return df

    @property
    def team_playing(self):
        try:
            playing = pd.concat([self._team_standings,
                                 self._team_attendance,
                                 self._team_batting,
                                 self._team_pitching,
                                 self._team_fielding],
                                ignore_index=True)
        except ValueError as exc:
            if "No objects" in str(exc):
                return None
            else:
                raise
        columns = [ 'league.year', 'league.name',
                    'entry.name', 'phase.name', 'division.name',
                    'R_G', 'R_W', 'R_L', 'R_T', 'R_PCT', 'R_RANK', 'R_ATT',
                    'B_G', 'B_IP', 'B_AB', 'B_R', 'B_ER', 'B_H', 'B_TB',
                    'B_1B', 'B_2B', 'B_3B', 'B_HR', 'B_RBI',
                    'B_BB', 'B_IBB', 'B_SO', 'B_GDP', 'B_HP',
                    'B_SH', 'B_SF', 'B_SB', 'B_CS',
                    'B_AVG',
                    'P_G', 'P_CG', 'P_SHO', 'P_GF',
                    'P_W', 'P_L', 'P_T', 'P_PCT',
                    'P_IP', 'P_AB', 'P_R', 'P_ER', 'P_H', 'P_HR',
                    'P_BB', 'P_IBB', 'P_SO', 'P_HP', 'P_SH', 'P_SF',
                    'P_WP', 'P_BK', 'P_ERA',
                    'F_G', 'F_TC', 'F_PO', 'F_A', 'F_E', 'F_DP', 'F_TP',
                    'F_PB', 'F_XI', 'F_LOB', 'F_PCT' ]
        return self._standardize_columns(playing, columns)

def defloat_columns(df):
    """Convert columns which should be integers to strings.  This deals with
    pandas' usage of floats for numeric columns which can have nulls."""
    df['league.year'] = df['league.year'].apply(int)
    for col in [x for x in df.columns
                if (x[:2] in ["B_", "F_", "P_", "M_", "R_"] and
                    x not in ["B_AVG", "P_IP", "P_ERA"] and
                    x[-4:] != "_PCT") or
                    (x in ["S_FIRST", "S_LAST", "seq"])]:
        df[col] = df[col].apply(lambda x: str(int(x)) if not pd.isnull(x)
                                else x)
    return df

def process_source(source):
    """Process workbooks from 'source', transforming all data and
    outputting to CSV files in processed.
    """
    books = [fn for fn in glob.glob("transcript/%s/*.xls" % source)
             if not "~" in fn]
    print "Processing source %s" % source
    for book in books:
        print "  %s" % book
    xlsbooks = [Workbook(fn) for fn in books]

    try:
        os.makedirs("processed/%s" % source)
    except os.error:
        pass

    ind_playing = pd.concat([book.individual_playing for book in xlsbooks],
                            ignore_index=True)
    ind_playing = defloat_columns(ind_playing)
    ind_playing.to_csv("processed/%s/playing_individual.csv" % source,
                       index=False)

    ind_managing = pd.concat([book.individual_managing for book in xlsbooks],
                             ignore_index=True)
    ind_managing = defloat_columns(ind_managing)
    ind_managing.to_csv("processed/%s/managing_individual.csv" % source,
                        index=False)

    try:
        team_playing = pd.concat([ent
                                  for ent in [book.team_playing
                                              for book in xlsbooks]
                                  if ent is not None],
                                 ignore_index=True)
        team_playing = defloat_columns(team_playing)
        team_playing.to_csv("processed/%s/playing_team.csv" % source, index=False)
    except ValueError as exc:
        if "No objects to concatenate" not in str(exc):
            raise

    print

if __name__ == '__main__':
    import glob
    import sys
    import os

    process_source(sys.argv[1])
