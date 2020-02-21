import json
import pathlib
from collections import OrderedDict

import numpy as np
import pandas as pd


def dropnull(rec):
    if not isinstance(rec, dict):
        return rec
    return {
        k: dropnull(v)
        for (k, v) in rec.items()
        if not pd.isnull(v) and dropnull(v)
    }


def extract_club_splits(df, prefix):
    def split_games(x):
        if pd.isnull(x):
            return x
        if "@" in x:
            return x.split("@")[0]
        else:
            return None

    i = 1
    while True:
        colname = f"club{i}_name"
        if colname not in df:
            return df
        df.insert(loc=df.columns.get_loc(colname)+1,
                  column=f"club{i}_{prefix}_G",
                  value=df[colname].apply(split_games))
        df[colname] = df[colname].str.split("@").str[-1]
        i += 1


def format_percentages(df):
    def format_era(x):
        if pd.isnull(x):
            return x
        if "." in x:
            full, frac = x.split(".")
        else:
            full, frac = x, ""
        frac = frac.ljust(2, "0")
        return ".".join([full, frac])

    for col in ["totals_R_PCT", "totals_B_AVG", "totals_B_SLG",
                "totals_P_AVG", "totals_P_PCT",
                "totals_F_PCT", "totals_F_UT_PCT"]:
        if col in df:
            df[col] = (
                df[col].replace("1", "1.000").replace("0", "0.000")
                .str.ljust(5, "0").str.lstrip("0")
            )
    if "P_ERA" in df:
        df["P_ERA"] = df["P_ERA"].apply(format_era)
    return df


def format_dates(df):
    def format_date(x, season):
        if pd.isnull(x):
            return x
        if len(x) == 8:
            return f"{x[:4]}-{x[4:6]}-{x[6:]}"
        elif len(x) == 4:
            return f"{season}-{x[:2]}-{x[2:]}"
        elif len(x) == 2:
            return f"{season}-{x}"
        raise ValueError(f"Invalid date field {x}")

    for col in df:
        if col.endswith(("_FIRST", "_LAST")):
            df[col] = df[col].apply(lambda x:
                                    format_date(x,
                                                df.loc[0]["league_season"]))
    return df


def format_names(df):
    for col in ["name_last", "name_first"]:
        if col in df:
            df[col] = df[col].str.replace(chr(8220), '"')
            df[col] = df[col].str.replace(chr(8221), '"')
            df[col] = df[col].str.replace(chr(8217), "'")
    return df


def add_row_metadata(df, table):
    df.insert(loc=0, column='_table', value=table)
    df.insert(loc=1, column='_row', value=np.arange(len(df))+1)
    return df


def rename_columns(df, column_map):
    unknown = [c for c in df.columns if c not in column_map.keys()]
    if unknown:
        print(f"WARNING: Unknown columns {unknown}")
    df = df.rename(columns=column_map)
    if "game_type" not in df:
        df.insert(loc=df.columns.get_loc("league_name")+1,
                  column="game_type", value="regular")
    else:
        col = df["game_type"]
        df = df.drop(labels=["game_type"], axis=1)
        df.insert(loc=df.columns.get_loc("league_name")+1,
                  column="game_type", value=col)
    return df


def extract_standings_team(df):
    column_map = {
        "year":            "league_season",
        "nameLeague":      "league_name",
        "nameClub":        "name_short",
        "division":        "division_name",
        "phase":           "game_type",
        "G":               "totals_R_G",
        "W":               "totals_R_W",
        "L":               "totals_R_L",
        "T":               "totals_R_T",
        "PCT":             "totals_R_PCT",
        "RANK":            "totals_R_RANK",
        "NOTES":           "NOTES",
    }
    df = (
        df.pipe(rename_columns, column_map)
        .pipe(add_row_metadata, "standings")
        .pipe(format_percentages)
        .pipe(format_dates)
        .pipe(transform_team_name)
        .pipe(transform_totals)
        .pipe(transform_team_playing)
    )
    return {
        "teams":
        [dropnull(x)
         for x in
         df[['_table', '_row', 'name', 'playing']].to_dict(orient='records')]
    }


def extract_head_to_head(df):
    # TODO: Format for head-to-head records
    return {}
    df = pd.melt(df, id_vars=["year", "nameLeague", "nameClub"],
                 var_name="opponent_name", value_name="R_W")
    column_map = {
        "year":            "league_season",
        "nameLeague":      "league_name",
        "nameClub":        "club_name",
        "phase":           "game_type",
        "opponent_name":   "opponent_name",
        "R_W":             "R_W",
    }
    df = (
        df[~df["R_W"].isnull()]
        .pipe(rename_columns, column_map)
        .pipe(add_row_metadata, "headtohead_team", "playing_team")
        .pipe(format_percentages)
        .pipe(format_dates)
    )
    return [dropnull(x) for x in df.to_dict(orient='records')]


def extract_attendance_team(df):
    column_map = {
        "year":            "league_season",
        "nameLeague":      "league_name",
        "nameClub":        "name_short",
        "phase":           "game_type",
        "ATT":             "totals_R_ATT",
    }
    df = (
        df.pipe(rename_columns, column_map)
        .pipe(add_row_metadata, "attendance")
        .pipe(format_percentages)
        .pipe(format_dates)
        .pipe(transform_team_name)
        .pipe(transform_totals)
        .pipe(transform_team_playing)
    )
    return {
        "teams":
        [dropnull(x)
         for x in
         df[['_table', '_row', 'name', 'playing']].to_dict(orient='records')]
    }


def transform_team_name(df):
    df['name'] = (
        df.filter(like="name_", axis=1)
        .rename(mapper=lambda x: x.replace("name_", ""), axis='columns')
        .apply(lambda x: dropnull(x.to_dict()), axis=1)
    )
    return df


def transform_team_playing(df):
    df['playing'] = (
        df.apply(lambda x: [{'season': x['league_season'],
                             'league': {'name': x['league_name']},
                             'game_type': x['game_type'],
                             'totals': x['totals']}], axis=1)
    )
    return df


def extract_batting_team(df):
    column_map = {
        "year":            "league_season",
        "nameLeague":      "league_name",
        "nameClub":        "name_short",
        "phase":           "game_type",
        "G":               "totals_B_G",
        "IP":              "totals_B_IP",    # team innings batted - rare
        "AB":              "totals_B_AB",
        "R":               "totals_B_R",
        "OR":              "totals_P_R",
        "ER":              "totals_B_ER",
        "H":               "totals_B_H",
        "TB":              "totals_B_TB",
        "EB":              "totals_B_EB",    # "extra bases"
        "H1B":             "totals_B_1B",
        "H2B":             "totals_B_2B",
        "H3B":             "totals_B_3B",
        "HR":              "totals_B_HR",
        "RBI":             "totals_B_RBI",
        "BB":              "totals_B_BB",
        "IBB":             "totals_B_IBB",
        "SO":              "totals_B_SO",
        "GDP":             "totals_B_GDP",
        "HP":              "totals_B_HP",
        "SH":              "totals_B_SH",
        "SF":              "totals_B_SF",
        "SB":              "totals_B_SB",
        "CS":              "totals_B_CS",
        "ROE":             "totals_B_ROE",
        "LOB":             "totals_B_LOB",
        "AVG":             "totals_B_AVG",
        "SLG":             "totals_B_SLG",
        "W":               "totals_R_W",
        "L":               "totals_R_L",
        "T":               "totals_R_T"
    }
    df = (
        df.pipe(rename_columns, column_map)
        .pipe(add_row_metadata, "batting")
        .pipe(format_percentages)
        .pipe(transform_team_name)
        .pipe(transform_totals)
        .pipe(transform_team_playing)
    )
    return {
        "teams":
        [dropnull(x)
         for x in
         df[['_table', '_row', 'name', 'playing']].to_dict(orient='records')]
    }


def extract_pitching_team(df):
    column_map = {
        "year":            "league_season",
        "nameLeague":      "league_name",
        "nameClub":        "name_short",
        "phase":           "game_type",
        "GP":              "totals_P_G",
        "APP":             "totals_P_APP",
        "GS":              "totals_P_GS",
        "CG":              "totals_P_CG",
        "SHO":             "totals_P_SHO",
        "GF":              "totals_P_GF",
        "W":               "totals_P_W",
        "L":               "totals_P_L",
        "T":               "totals_P_T",
        "PCT":             "totals_P_PCT",
        "IP":              "totals_P_IP",
        "AB":              "totals_P_AB",
        "R":               "totals_P_R",
        "ER":              "totals_P_ER",
        "H":               "totals_P_H",
        "HR":              "totals_P_HR",
        "BB":              "totals_P_BB",
        "IBB":             "totals_P_IBB",
        "SO":              "totals_P_SO",
        "HB":              "totals_P_HP",
        "SH":              "totals_P_SH",
        "SF":              "totals_P_SF",
        "WP":              "totals_P_WP",
        "BK":              "totals_P_BK",
        "SB":              "totals_P_SB",
        "CS":              "totals_P_CS",
        "ERA":             "totals_P_ERA",
    }
    df = (
        df.pipe(rename_columns, column_map)
        .pipe(add_row_metadata, "pitching")
        .pipe(format_percentages)
        .pipe(format_dates)
        .pipe(transform_team_name)
        .pipe(transform_totals)
        .pipe(transform_team_playing)
    )
    return {
        "teams":
        [dropnull(x)
         for x in
         df[['_table', '_row', 'name', 'playing']].to_dict(orient='records')]
    }


def extract_fielding_team(df):
    column_map = {
        "year":            "league_season",
        "nameLeague":      "league_name",
        "nameClub":        "name_short",
        "phase":           "game_type",
        "G":               "totals_F_G",
        "TC":              "totals_F_TC",
        "PO":              "totals_F_PO",
        "A":               "totals_F_A",
        "E":               "totals_F_E",
        "DP":              "totals_F_DP",
        "TP":              "totals_F_TP",
        "PB":              "totals_F_PB",
        "SB":              "totals_F_SB",
        "CS":              "totals_F_CS",
        "PCT":             "totals_F_PCT",
        "P_W":             "totals_P_W",
        "P_L":             "totals_P_L",
        "P_T":             "totals_P_T",
        "LOB":             "totals_P_LOB",
    }
    df = (
        df.pipe(rename_columns, column_map)
        .pipe(add_row_metadata, "fielding")
        .pipe(format_percentages)
        .pipe(format_dates)
        .pipe(transform_team_name)
        .pipe(transform_totals)
        .pipe(transform_team_playing)
    )
    return {
        "teams":
        [dropnull(x)
         for x in
         df[['_table', '_row', 'name', 'playing']].to_dict(orient='records')]
    }


def extract_managing_individual(df):
    column_map = {
        "year":            "league_season",
        "nameLeague":      "league_name",
        "nameLast":        "name_last",
        "nameFirst":       "name_first",
        "nameClub":        "club1_name",
        "seq":             "club1_S_ORDER",
        "dateFirst":       "club1_S_FIRST",
        "dateLast":        "club1_S_LAST",
    }
    df = (
        df.pipe(rename_columns, column_map)
        .pipe(add_row_metadata, "managing")
        .pipe(extract_club_splits, "M")
        .pipe(format_percentages)
        .pipe(format_dates)
        .pipe(format_names)
        .pipe(transform_person_name)
        .pipe(transform_person_description)
        .pipe(transform_person_club_splits, prefix="M")
        .pipe(transform_totals)
        .pipe(transform_person_managing)
    )
    return {
        "people":
        [dropnull(x)
         for x in
         df[['_table', '_row',
             'name', 'description', 'managing']].to_dict(orient='records')]
    }


def extract_umpiring_individual(df):
    column_map = {
        "year":            "league_season",
        "nameLeague":      "league_name",
        "nameLast":        "person_name_last",
        "nameFirst":       "person_name_first",
        "dateFirst":       "S_FIRST",
        "dateLast":        "S_LAST",
    }
    df = (
        df.pipe(rename_columns, column_map)
        .pipe(add_row_metadata, "umpiring_individual", "umpiring_individual")
        .pipe(format_percentages)
        .pipe(format_dates)
        .pipe(format_names)
    )
    return [dropnull(x) for x in df.to_dict(orient='records')]


def transform_person_name(df):
    df['name'] = (
        df.filter(like="name_", axis=1)
        .rename(mapper=lambda x: x.replace("name_", ""), axis='columns')
        .apply(lambda x: dropnull(x.to_dict()), axis=1)
    )
    return df


def transform_person_description(df):
    df['description'] = (
        df.filter(like="description_", axis=1)
        .rename(mapper=lambda x: x.replace("description_", ""), axis='columns')
        .apply(lambda x: dropnull(x.to_dict()), axis=1)
    )
    return df

def transform_person_club_splits(df, prefix):
    for i in [1, 2, 3, 4, 5]:
        if f'club{i}_name' not in df:
            continue
        df[f'split_{i}'] = (
            df.apply(lambda x:
                     dropnull(
                         {'team': dropnull({'name': x[f'club{i}_name']}),
                          'S_ORDER':
                          x[f'club{i}_S_ORDER'] if f'club{i}_S_ORDER' in x
                          else None,
                          'S_FIRST':
                          x[f'club{i}_S_FIRST'] if f'club{i}_S_FIRST' in x
                          else None,
                          'S_LAST':
                          x[f'club{i}_S_LAST'] if f'club{i}_S_LAST' in x
                          else None,
                          f'{prefix}_G': x[f'club{i}_{prefix}_G']}
                     ),
                     axis=1)
        )
    df['splits'] = (
        df.apply(lambda x: [x[c]
                            for c in df.columns
                            if c.startswith('split_') and x[c]],
                 axis=1)
    )
    return df

def transform_totals(df):
    df['totals'] = (
        df.filter(like="totals_", axis=1)
        .rename(mapper=lambda x: x.replace("totals_", ""), axis='columns')
        .apply(lambda x: dropnull(x.to_dict()), axis=1)
    )
    return df


def transform_person_playing(df):
    df['playing'] = (
        df.apply(lambda x: [{'season': x['league_season'],
                             'league': {'name': x['league_name']},
                             'splits': x['splits'],
                             'totals': x['totals']}], axis=1)
    )
    return df


def transform_person_managing(df):
    df['managing'] = (
        df.apply(lambda x: [
            dropnull({'season': x['league_season'],
                      'league': {'name': x['league_name']},
                      'splits': x['splits'],
                      'totals': x['totals']})], axis=1)
    )
    return df


def extract_batting_individual(df):
    column_map = {
        "year":            "league_season",
        "nameLeague":      "league_name",
        "nameLast":        "name_last",
        "nameFirst":       "name_first",
        "bats":            "description_bats",
        "throws":          "description_throws",
        "nameClub":        "club1_name",
        "nameClub1":       "club1_name",
        "nameClub2":       "club2_name",
        "nameClub3":       "club3_name",
        "nameClub4":       "club4_name",
        "nameClub5":       "club5_name",
        "phase":           "game_type",
        "S_STINT":         "S_STINT",
        "dateFirst":       "S_FIRST",
        "dateLast":        "S_LAST",
        "Pos":             "totals_F_POS",
        "G":               "totals_B_G",
        "AB":              "totals_B_AB",
        "R":               "totals_B_R",
        "ER":              "totals_B_ER",
        "H":               "totals_B_H",
        "TB":              "totals_B_TB",
        "EB":              "totals_B_EB",    # "extra bases"
        "H1B":             "totals_B_1B",
        "H2B":             "totals_B_2B",
        "H3B":             "totals_B_3B",
        "HR":              "totals_B_HR",
        "RBI":             "totals_B_RBI",
        "BB":              "totals_B_BB",
        "IBB":             "totals_B_IBB",
        "SO":              "totals_B_SO",
        "GDP":             "totals_B_GDP",
        "HP":              "totals_B_HP",
        "SH":              "totals_B_SH",
        "SF":              "totals_B_SF",
        "SB":              "totals_B_SB",
        "CS":              "totals_B_CS",
        "AVG":             "totals_B_AVG",
        "AVG_RANK":        "totals_B_AVG_RANK",
        "SLG":             "totals_B_SLG",
        "F_P_G":           "F_P_G",
        "F_C_G":           "F_C_G",
        "F_1B_G":          "F_1B_G",
        "F_2B_G":          "F_2B_G",
        "F_3B_G":          "F_3B_G",
        "F_SS_G":          "F_SS_G",
        "F_OF_G":          "F_OF_G",
        "F_LF_G":          "F_LF_G",
        "F_CF_G":          "F_CF_G",
        "F_RF_G":          "F_RF_G",
        "B_G_PH":          "B_G_PH",
        "B_G_PR":          "B_G_PR",
        "NOTES":           "NOTES",
    }
    df = (
        df.pipe(rename_columns, column_map)
        .pipe(add_row_metadata, "batting")
        .pipe(extract_club_splits, "B")
        .pipe(format_percentages)
        .pipe(format_dates)
        .pipe(format_names)
    )
    df['club1_name'] = df['club1_name'].replace({"all": None})

    df = (
        df.pipe(transform_person_name)
        .pipe(transform_person_description)
        .pipe(transform_person_club_splits, prefix="B")
        .pipe(transform_totals)
        .pipe(transform_person_playing)
    )
    return {
        "people":
        [dropnull(x)
         for x in
         df[['_table', '_row',
             'name', 'description', 'playing']].to_dict(orient='records')]
    }


def extract_pitching_individual(df):
    column_map = {
        "year":            "league_season",
        "nameLeague":      "league_name",
        "nameLast":        "name_last",
        "nameFirst":       "name_first",
        "throws":          "description_throws",
        "nameClub":        "club1_name",
        "nameClub1":       "club1_name",
        "nameClub2":       "club2_name",
        "nameClub3":       "club3_name",
        "nameClub4":       "club4_name",
        "nameClub5":       "club5_name",
        "phase":           "game_type",
        "GP":              "totals_P_G",
        "GS":              "totals_P_GS",
        "REL":             "totals_P_G_RP",    # games as reliever
        "EIG":             "totals_P_G_EI",    # extra-inning games
        "0H":              "totals_P_G_0H",
        "1H":              "totals_P_G_1H",
        "2H":              "totals_P_G_2H",
        "3H":              "totals_P_G_3H",
        "4H":              "totals_P_G_4H",
        "5H":              "totals_P_G_5H",
        "CG":              "totals_P_CG",
        "SHO":             "totals_P_SHO",
        "TO":              "totals_P_TO",
        "GF":              "totals_P_GF",
        "DEC":             "totals_P_DEC",
        "W":               "totals_P_W",
        "L":               "totals_P_L",
        "T":               "totals_P_T",
        "ND":              "totals_P_ND",
        "PCT":             "totals_P_PCT",
        "IP":              "totals_P_IP",
        "TBF":             "totals_P_TBF",
        "AB":              "totals_P_AB",
        "R":               "totals_P_R",
        "R/G":             "totals_P_RPG",    # runs per game
        "ER":              "totals_P_ER",
        "H":               "totals_P_H",
        "H/G":             "totals_P_HPG",    # hits per game
        "TB":              "totals_P_TB",
        "H2B":             "totals_P_2B",
        "H3B":             "totals_P_3B",
        "HR":              "totals_P_HR",
        "BB":              "totals_P_BB",
        "IBB":             "totals_P_IBB",
        "SO":              "totals_P_SO",
        "HB":              "totals_P_HP",
        "SH":              "totals_P_SH",
        "WP":              "totals_P_WP",
        "BK":              "totals_P_BK",
        "SB":              "totals_P_SB",
        "AVG":             "totals_P_AVG",
        "ERA":             "totals_P_ERA",
        "ERA_RANK":        "totals_P_ERA_RANK",
        "NOTES":           "NOTES",
    }
    df = (
        df.pipe(rename_columns, column_map)
        .pipe(add_row_metadata, "pitching")
        .assign(totals_F_P_POS="1")
        .pipe(extract_club_splits, "P")
        .pipe(format_percentages)
        .pipe(format_dates)
        .pipe(format_names)
        .pipe(transform_person_name)
        .pipe(transform_person_description)
        .pipe(transform_person_club_splits, prefix="P")
        .pipe(transform_totals)
        .pipe(transform_person_playing)
    )
    return {
        "people":
        [dropnull(x)
         for x in
         df[['_table', '_row',
             'name', 'description', 'playing']].to_dict(orient='records')]
    }


def extract_fielding_individual(df):
    column_map = {
        "year":            "league_season",
        "nameLeague":      "league_name",
        "nameLast":        "name_last",
        "nameFirst":       "name_first",
        "throws":          "description_throws",
        "nameClub":        "club1_name",
        "nameClub1":       "club1_name",
        "nameClub2":       "club2_name",
        "nameClub3":       "club3_name",
        "nameClub4":       "club4_name",
        "nameClub5":       "club5_name",
        "phase":           "game_type",
        "Pos":             "totals_F_POS",
        "G":               "totals_F_G",
        "ALL_G":           "totals_F_UT_G",
        "INN":             "totals_F_INN",
        "TC":              "totals_F_TC",
        "PO":              "totals_F_PO",
        "ALL_PO":          "totals_F_UT_PO",
        "A":               "totals_F_A",
        "ALL_A":           "totals_F_UT_A",
        "E":               "totals_F_E",
        "ALL_E":           "totals_F_UT_E",
        "DP":              "totals_F_DP",
        "ALL_DP":          "totals_F_UT_DP",
        "TP":              "totals_F_TP",
        "PB":              "totals_F_PB",
        "SB":              "totals_F_SB",
        "CS":              "totals_F_CS",
        "CN":              "totals_F_PK",
        "PCT":             "totals_F_PCT",
        "ALL_PCT":         "totals_F_UT_PCT",
        "P_WP":            "totals_P_WP",
        "NOTES":           "NOTES",
    }
    df = (
        df.pipe(rename_columns, column_map)
        .pipe(add_row_metadata, "fielding")
        .pipe(extract_club_splits, "F")
        .pipe(format_percentages)
        .pipe(format_dates)
        .pipe(format_names)
        .pipe(transform_person_name)
        .pipe(transform_person_description)
        .pipe(transform_person_club_splits, prefix="F")
        .pipe(transform_totals)
        .pipe(transform_person_playing)
    )
    return {
        "people":
        [dropnull(x)
         for x in
         df[['_table', '_row',
             'name', 'description', 'playing']].to_dict(orient='records')]
    }


function_map = {
    "Batting":       extract_batting_individual,
    "Pitching":      extract_pitching_individual,
    "Fielding":      extract_fielding_individual,
    "TeamBatting":   extract_batting_team,
    "TeamPitching":  extract_pitching_team,
    "TeamFielding":  extract_fielding_team,
    "Standings":     extract_standings_team,
    "HeadToHead":    extract_head_to_head,
    "Attendance":    extract_attendance_team,
    "Managing":      extract_managing_individual,
#    "Umpiring":      extract_umpiring_individual,
}


def process_file(source, fn):
    data = OrderedDict()
    data["_source"] = OrderedDict()
    data["_source"]["title"] = source
    data["teams"] = []
    data["people"] = []
    for (name, df) in pd.read_excel(fn,
                                    dtype=str, sheet_name=None).items():
        if name == "Metadata":
            continue
        if name not in function_map:
            print(f"WARNING: Unknown sheet name {name}")
            continue
        print(f"Processing worksheet {name}")
        result = function_map[name](df)
        for key in ["people", "teams"]:
            try:
                data[key].extend(result[key])
            except KeyError:
                pass
    return data


def main(source):
    inpath = pathlib.Path("transcript")/source
    outpath = pathlib.Path("json")
    outpath.mkdir(exist_ok=True, parents=True)

    books = []
    for fn in sorted(inpath.glob("*.xls")):
        print(f"Processing {fn}")
        books.append(process_file(source, fn))
        print()
        break

    js = json.dumps(books, indent=2)
    with (outpath / f"{source}.json").open("w") as f:
            f.write(js)
    print()
