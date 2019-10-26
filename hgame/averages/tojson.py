import sys
import json
from collections import OrderedDict

import numpy as np
import pandas as pd


def dropnull(rec):
    return {
        k: v
        for (k, v) in rec.items()
        if isinstance(v, list) or not pd.isnull(v)
    }


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

    for col in ["R_PCT", "B_AVG", "P_AVG", "P_PCT", "F_PCT"]:
        if col in df:
            df[col] = (
                df[col].replace("1", "1.000").replace("0", "0.000")
                .str.ljust(5, "0").str.lstrip("0")
            )
    if "P_ERA" in df:
        df["P_ERA"] = df["P_ERA"].apply(format_era)
    return df
    

def add_row_metadata(df, record_type):
    df.insert(loc=0, column='_row', value=np.arange(len(df))+1)
    df.insert(loc=1, column='_type', value=record_type)
    return df

def rename_columns(df, column_map):
    unknown = [c for c in df.columns if c not in column_map.keys()]
    if unknown:
        print(f"WARNING: Unknown columns {unknown}")
    df = df.rename(columns=column_map)
    if "league_phase" not in df:
        df.insert(loc=df.columns.get_loc("league_name")+1,
                  column="league_phase", value="regular")
    else:
        col = df["league_phase"]
        df = df.drop(labels=["league_phase"], axis=1)
        df.insert(loc=df.columns.get_loc("league_name")+1,
                  column="league_phase", value=col)
    return df


def extract_head_to_head(df):
    df = pd.melt(df, id_vars=["year", "nameLeague", "nameClub"],
                 var_name="opponent_name", value_name="R_W")
    column_map = {
        "year":            "league_season",
        "nameLeague":      "league_name",
        "nameClub":        "club_name",
        "phase":           "league_phase",
        "opponent_name":   "opponent_name",
        "R_W":             "R_W",
    }
    df = df[~df["R_W"].isnull()].pipe(rename_columns, column_map)
    df = df.pipe(add_row_metadata, "playing_team").pipe(format_percentages)
    return [dropnull(x) for x in df.to_dict(orient='records')]
    

def extract_standings_team(df):
    column_map = {
        "year":            "league_season",
        "nameLeague":      "league_name",
        "nameClub":        "club_name",
        "phase":           "league_phase",
        "G":               "R_G",
        "W":               "R_W",
        "L":               "R_L",
        "T":               "R_T",
        "PCT":             "R_PCT",
        "RANK":            "R_RANK",
        "NOTES":           "NOTES",
    }
    df = df.pipe(rename_columns, column_map)
    df = df.pipe(add_row_metadata, "playing_team").pipe(format_percentages)
    return [dropnull(x) for x in df.to_dict(orient='records')]
        

def extract_batting_team(df):
    column_map = {
        "year":            "league_season",
        "nameLeague":      "league_name",
        "nameClub":        "club_name",
        "G":               "B_G",
        "AB":              "B_AB",
        "R":               "B_R",
        "ER":              "B_ER",
        "H":               "B_H",
        "TB":              "B_TB",
        "H1B":             "B_1B",
        "H2B":             "B_2B",
        "H3B":             "B_3B",
        "HR":              "B_HR",
        "BB":              "B_BB",
        "SO":              "B_SO",
        "SH":              "B_SH",
        "SB":              "B_SB",
        "LOB":             "B_LOB",
        "AVG":             "B_AVG",
        "W":               "R_W",
        "L":               "R_L",
        "T":               "R_T"
    }
    df = df.pipe(rename_columns, column_map)
    df = df.pipe(add_row_metadata, "playing_team").pipe(format_percentages)
    return [dropnull(x) for x in df.to_dict(orient='records')]
    
def extract_fielding_team(df):
    column_map = {
        "year":            "league_season",
        "nameLeague":      "league_name",
        "nameClub":        "club_name",
        "G":               "F_G",
        "TC":              "F_TC",
        "PO":              "F_PO",
        "A":               "F_A",
        "E":               "F_E",
        "DP":              "F_DP",
        "TP":              "F_TP",
        "PB":              "F_PB",
        "PCT":             "F_PCT",
        "P_W":             "P_W",
        "P_L":             "P_L",
        "P_T":             "P_T",
    }
    df = df.pipe(rename_columns, column_map)
    df = df.pipe(add_row_metadata, "playing_team").pipe(format_percentages)
    return [dropnull(x) for x in df.to_dict(orient='records')]

def extract_batting_individual(df):
    column_map = {
        "year":            "league_season",
        "nameLeague":      "league_name",
        "nameLast":        "person_name_last",
        "nameFirst":       "person_name_first",
        "nameClub":        "club1_name",
        "nameClub1":       "club1_name",
        "nameClub2":       "club2_name",
        "nameClub3":       "club3_name",
        "nameClub4":       "club4_name",
        "G":               "B_G",
        "AB":              "B_AB",
        "R":               "B_R",
        "ER":              "B_ER",
        "H":               "B_H",
        "TB":              "B_TB",
        "EB":              "B_EB",    # "extra bases"
        "H1B":             "B_1B",
        "H2B":             "B_2B",
        "H3B":             "B_3B",
        "HR":              "B_HR",
        "BB":              "B_BB",
        "SO":              "B_SO",
        "SH":              "B_SH",
        "SB":              "B_SB",
        "AVG":             "B_AVG",
        "NOTES":           "NOTES",
    }
    df = df.pipe(rename_columns, column_map)
    df = df.pipe(add_row_metadata, "playing_individual").pipe(format_percentages)
    return [dropnull(x) for x in df.to_dict(orient='records')]


def extract_pitching_individual(df):
    column_map = {
        "year":            "league_season",
        "nameLeague":      "league_name",
        "nameLast":        "person_name_last",
        "nameFirst":       "person_name_first",
        "nameClub":        "club1_name",
        "nameClub1":       "club1_name",
        "nameClub2":       "club2_name",
        "nameClub3":       "club3_name",
        "nameClub4":       "club4_name",
        "GP":              "P_G",
        "RL":              "P_G_RP",    # games as reliever
        "CG":              "P_CG",
        "SHO":             "P_SHO",
        "TO":              "P_TO",
        "GF":              "P_GF",
        "W":               "P_W",
        "L":               "P_L",
        "T":               "P_T",
        "PCT":             "P_PCT",
        "IP":              "P_IP",
        "TBF":             "P_TBF",
        "AB":              "P_AB",
        "R":               "P_R",
        "ER":              "P_ER",
        "H":               "P_H",
        "BB":              "P_BB",
        "SO":              "P_SO",
        "HB":              "P_HP",
        "SH":              "P_SH",
        "WP":              "P_WP",
        "BK":              "P_BK",
        "SB":              "P_SB",
        "AVG":             "P_AVG",
        "ERA":             "P_ERA",
        "NOTES":           "NOTES",
    }
    df = df.pipe(rename_columns, column_map)
    df = df.pipe(add_row_metadata, "playing_individual").pipe(format_percentages)
    return [dropnull(x) for x in df.to_dict(orient='records')]


def extract_fielding_individual(df):
    column_map = {
        "year":            "league_season",
        "nameLeague":      "league_name",
        "nameLast":        "person_name_last",
        "nameFirst":       "person_name_first",
        "nameClub":        "club1_name",
        "nameClub1":       "club1_name",
        "nameClub2":       "club2_name",
        "nameClub3":       "club3_name",
        "nameClub4":       "club4_name",
        "Pos":             "F_POS",
        "G":               "F_G",
        "INN":             "F_INN",
        "PO":              "F_PO",
        "A":               "F_A",
        "E":               "F_E",
        "TC":              "F_TC",
        "PB":              "F_PB",
        "SB":              "F_SB",
        "CS":              "F_CS",
        "CN":              "F_PK",
        "PCT":             "F_PCT",
        "NOTES":           "NOTES",
    }
    df = df.pipe(rename_columns, column_map)
    df = df.pipe(add_row_metadata, "playing_individual").pipe(format_percentages)
    return [dropnull(x) for x in df.to_dict(orient='records')]


function_map = {
    "Batting":   extract_batting_individual,
    "Pitching":  extract_pitching_individual,
    "Fielding":  extract_fielding_individual,
    "TeamBatting":   extract_batting_team,
    "TeamFielding":  extract_fielding_team,
    "Standings":    extract_standings_team,
    "HeadToHead":   extract_head_to_head,
}

name_map = {
    "Batting":   "batting_individual",
    "Pitching":  "pitching_individual",
    "Fielding":  "fielding_individual",
    "TeamBatting":   "batting_team",
    "TeamFielding":  "fielding_team",
    "Standings":     "standings_team",
    "HeadToHead":    "headtohead_team",
}

def main():
    leagues = []
    tables = OrderedDict()
    for (name, data) in pd.read_excel(sys.argv[1],
                                      dtype=str, sheet_name=None).items():
        if name not in function_map:
            print(f"WARNING: Unknown sheet name {name}")
            continue
        print(f"Processing worksheet {name}")
        leagues.append(data[['year', 'nameLeague']].drop_duplicates())
        tables[name_map[name]] = function_map[name](data)

    js = json.dumps(tables, indent=2)
    with open("outfile.json", "w") as f:
        f.write(js)


if __name__ == '__main__':
    main()
