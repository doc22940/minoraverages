import json
import pathlib
from collections import OrderedDict

import numpy as np
import pandas as pd


def dropnull(rec):
    return {
        k: v
        for (k, v) in rec.items()
        if isinstance(v, list) or not pd.isnull(v)
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

    for col in ["R_PCT", "B_AVG", "B_SLG", "P_AVG", "P_PCT",
                "F_PCT", "F_UT_PCT"]:
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

    for col in ["S_FIRST", "S_LAST"]:
        if col in df:
            df[col] = df[col].apply(lambda x:
                                    format_date(x,
                                                df.loc[0]["league_season"]))
    return df


def format_names(df):
    for col in ["person_name_last", "person_name_first"]:
        if col in df:
            df[col] = df[col].str.replace(chr(8220), '"')
            df[col] = df[col].str.replace(chr(8221), '"')
            df[col] = df[col].str.replace(chr(8217), "'")
    return df


def add_row_metadata(df, table, record_type):
    df.insert(loc=0, column='_table', value=table)
    df.insert(loc=1, column='_row', value=np.arange(len(df))+1)
    df.insert(loc=2, column='_type', value=record_type)
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


def extract_standings_team(df):
    column_map = {
        "year":            "league_season",
        "nameLeague":      "league_name",
        "nameClub":        "club_name",
        "division":        "division_name",
        "phase":           "league_phase",
        "G":               "R_G",
        "W":               "R_W",
        "L":               "R_L",
        "T":               "R_T",
        "PCT":             "R_PCT",
        "RANK":            "R_RANK",
        "NOTES":           "NOTES",
    }
    df = (
        df.pipe(rename_columns, column_map)
        .pipe(add_row_metadata, "standings_team", "playing_team")
        .pipe(format_percentages)
        .pipe(format_dates)
    )
    return [dropnull(x) for x in df.to_dict(orient='records')]


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
        "nameClub":        "club_name",
        "phase":           "league_phase",
        "ATT":             "R_ATT",
    }
    df = (
        df.pipe(rename_columns, column_map)
        .pipe(add_row_metadata, "attendance_team", "playing_team")
        .pipe(format_percentages)
        .pipe(format_dates)
    )
    return [dropnull(x) for x in df.to_dict(orient='records')]


def extract_batting_team(df):
    column_map = {
        "year":            "league_season",
        "nameLeague":      "league_name",
        "nameClub":        "club_name",
        "phase":           "league_phase",
        "G":               "B_G",
        "IP":              "B_IP",    # team innings batted - rare
        "AB":              "B_AB",
        "R":               "B_R",
        "OR":              "P_R",
        "ER":              "B_ER",
        "H":               "B_H",
        "TB":              "B_TB",
        "EB":              "B_EB",    # "extra bases"
        "H1B":             "B_1B",
        "H2B":             "B_2B",
        "H3B":             "B_3B",
        "HR":              "B_HR",
        "RBI":             "B_RBI",
        "BB":              "B_BB",
        "IBB":             "B_IBB",
        "SO":              "B_SO",
        "GDP":             "B_GDP",
        "HP":              "B_HP",
        "SH":              "B_SH",
        "SF":              "B_SF",
        "SB":              "B_SB",
        "CS":              "B_CS",
        "ROE":             "B_ROE",
        "LOB":             "B_LOB",
        "AVG":             "B_AVG",
        "SLG":             "B_SLG",
        "W":               "R_W",
        "L":               "R_L",
        "T":               "R_T"
    }
    df = (
        df.pipe(rename_columns, column_map)
        .pipe(add_row_metadata, "batting_team", "playing_team")
        .pipe(format_percentages)
        .pipe(format_dates)
    )
    return [dropnull(x) for x in df.to_dict(orient='records')]


def extract_pitching_team(df):
    column_map = {
        "year":            "league_season",
        "nameLeague":      "league_name",
        "nameClub":        "club_name",
        "phase":           "league_phase",
        "GP":              "P_G",
        "GS":              "P_GS",
        "CG":              "P_CG",
        "SHO":             "P_SHO",
        "GF":              "P_GF",
        "W":               "P_W",
        "L":               "P_L",
        "T":               "P_T",
        "PCT":             "P_PCT",
        "IP":              "P_IP",
        "AB":              "P_AB",
        "R":               "P_R",
        "ER":              "P_ER",
        "H":               "P_H",
        "HR":              "P_HR",
        "BB":              "P_BB",
        "IBB":             "P_IBB",
        "SO":              "P_SO",
        "HB":              "P_HP",
        "SH":              "P_SH",
        "SF":              "P_SF",
        "WP":              "P_WP",
        "BK":              "P_BK",
        "SB":              "P_SB",
        "CS":              "P_CS",
        "ERA":             "P_ERA",
    }
    df = (
        df.pipe(rename_columns, column_map)
        .pipe(add_row_metadata, "pitching_team", "playing_team")
        .pipe(format_percentages)
        .pipe(format_dates)
    )
    return [dropnull(x) for x in df.to_dict(orient='records')]


def extract_fielding_team(df):
    column_map = {
        "year":            "league_season",
        "nameLeague":      "league_name",
        "nameClub":        "club_name",
        "phase":           "league_phase",
        "G":               "F_G",
        "TC":              "F_TC",
        "PO":              "F_PO",
        "A":               "F_A",
        "E":               "F_E",
        "DP":              "F_DP",
        "TP":              "F_TP",
        "PB":              "F_PB",
        "SB":              "F_SB",
        "PCT":             "F_PCT",
        "P_W":             "P_W",
        "P_L":             "P_L",
        "P_T":             "P_T",
        "LOB":             "P_LOB",
    }
    df = (
        df.pipe(rename_columns, column_map)
        .pipe(add_row_metadata, "fielding_team", "playing_team")
        .pipe(format_percentages)
        .pipe(format_dates)
    )
    return [dropnull(x) for x in df.to_dict(orient='records')]


def extract_managing_individual(df):
    column_map = {
        "year":            "league_season",
        "nameLeague":      "league_name",
        "nameLast":        "person_name_last",
        "nameFirst":       "person_name_first",
        "nameClub":        "club_name",
        "seq":             "S_ORDER",
        "dateFirst":       "S_FIRST",
        "dateLast":        "S_LAST",
    }
    df = (
        df.pipe(rename_columns, column_map)
        .pipe(add_row_metadata, "managing_individual", "managing_individual")
        .pipe(format_percentages)
        .pipe(format_dates)
        .pipe(format_names)
    )
    return [dropnull(x) for x in df.to_dict(orient='records')]


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


def extract_batting_individual(df):
    column_map = {
        "year":            "league_season",
        "nameLeague":      "league_name",
        "nameLast":        "person_name_last",
        "nameFirst":       "person_name_first",
        "bats":            "person_bats",
        "throws":          "person_throws",
        "nameClub":        "club1_name",
        "nameClub1":       "club1_name",
        "nameClub2":       "club2_name",
        "nameClub3":       "club3_name",
        "nameClub4":       "club4_name",
        "nameClub5":       "club5_name",
        "phase":           "league_phase",
        "S_STINT":         "S_STINT",
        "dateFirst":       "S_FIRST",
        "dateLast":        "S_LAST",
        "Pos":             "F_POS",
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
        "RBI":             "B_RBI",
        "BB":              "B_BB",
        "IBB":             "B_IBB",
        "SO":              "B_SO",
        "GDP":             "B_GDP",
        "HP":              "B_HP",
        "SH":              "B_SH",
        "SF":              "B_SF",
        "SB":              "B_SB",
        "CS":              "B_CS",
        "AVG":             "B_AVG",
        "AVG_RANK":        "B_AVG_RANK",
        "SLG":             "B_SLG",
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
        .pipe(add_row_metadata, "batting_individual", "playing_individual")
        .pipe(extract_club_splits, "B")
        .pipe(format_percentages)
        .pipe(format_dates)
        .pipe(format_names)
    )
    df['club1_name'] = df['club1_name'].replace({"all": None})
    return [dropnull(x) for x in df.to_dict(orient='records')]


def extract_pitching_individual(df):
    column_map = {
        "year":            "league_season",
        "nameLeague":      "league_name",
        "nameLast":        "person_name_last",
        "nameFirst":       "person_name_first",
        "throws":          "person_throws",
        "nameClub":        "club1_name",
        "nameClub1":       "club1_name",
        "nameClub2":       "club2_name",
        "nameClub3":       "club3_name",
        "nameClub4":       "club4_name",
        "nameClub5":       "club5_name",
        "phase":           "league_phase",
        "GP":              "P_G",
        "GS":              "P_GS",
        "REL":             "P_G_RP",    # games as reliever
        "EIG":             "P_G_EI",    # extra-inning games
        "0H":              "P_G_0H",
        "1H":              "P_G_1H",
        "2H":              "P_G_2H",
        "3H":              "P_G_3H",
        "4H":              "P_G_4H",
        "5H":              "P_G_5H",
        "CG":              "P_CG",
        "SHO":             "P_SHO",
        "TO":              "P_TO",
        "GF":              "P_GF",
        "DEC":             "P_DEC",
        "W":               "P_W",
        "L":               "P_L",
        "T":               "P_T",
        "ND":              "P_ND",
        "PCT":             "P_PCT",
        "IP":              "P_IP",
        "TBF":             "P_TBF",
        "AB":              "P_AB",
        "R":               "P_R",
        "R/G":             "P_RPG",    # runs per gamey
        "ER":              "P_ER",
        "H":               "P_H",
        "H/G":             "P_HPG",    # hits per game
        "TB":              "P_TB",
        "H2B":             "P_2B",
        "H3B":             "P_3B",
        "HR":              "P_HR",
        "BB":              "P_BB",
        "IBB":             "B_IBB",
        "SO":              "P_SO",
        "HB":              "P_HP",
        "SH":              "P_SH",
        "WP":              "P_WP",
        "BK":              "P_BK",
        "SB":              "P_SB",
        "AVG":             "P_AVG",
        "ERA":             "P_ERA",
        "ERA_RANK":        "P_ERA_RANK",
        "NOTES":           "NOTES",
    }
    df = (
        df.pipe(rename_columns, column_map)
        .pipe(add_row_metadata, "pitching_individual", "playing_individual")
        .pipe(extract_club_splits, "P")
        .pipe(format_percentages)
        .pipe(format_dates)
        .pipe(format_names)
    )
    return [dropnull(x) for x in df.to_dict(orient='records')]


def extract_fielding_individual(df):
    column_map = {
        "year":            "league_season",
        "nameLeague":      "league_name",
        "nameLast":        "person_name_last",
        "nameFirst":       "person_name_first",
        "throws":          "person_throws",
        "nameClub":        "club1_name",
        "nameClub1":       "club1_name",
        "nameClub2":       "club2_name",
        "nameClub3":       "club3_name",
        "nameClub4":       "club4_name",
        "nameClub5":       "club5_name",
        "phase":           "league_phase",
        "Pos":             "F_POS",
        "G":               "F_G",
        "ALL_G":           "F_UT_G",
        "INN":             "F_INN",
        "TC":              "F_TC",
        "PO":              "F_PO",
        "ALL_PO":          "F_UT_PO",
        "A":               "F_A",
        "ALL_A":           "F_UT_A",
        "E":               "F_E",
        "ALL_E":           "F_UT_E",
        "DP":              "F_DP",
        "ALL_DP":          "F_UT_DP",
        "TP":              "F_TP",
        "PB":              "F_PB",
        "SB":              "F_SB",
        "CS":              "F_CS",
        "CN":              "F_PK",
        "PCT":             "F_PCT",
        "ALL_PCT":         "F_UT_PCT",
        "P_WP":            "P_WP",
        "NOTES":           "NOTES",
    }
    df = (
        df.pipe(rename_columns, column_map)
        .pipe(add_row_metadata, "fielding_individual", "playing_individual")
        .pipe(extract_club_splits, "F")
        .pipe(format_percentages)
        .pipe(format_dates)
        .pipe(format_names)
    )
    return [dropnull(x) for x in df.to_dict(orient='records')]


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
    "Umpiring":      extract_umpiring_individual,
}


def process_file(source, fn):
    data = OrderedDict()
    data["_source"] = OrderedDict()
    data["_source"]["title"] = source
    data["records"] = []
    for (name, df) in pd.read_excel(fn,
                                    dtype=str, sheet_name=None).items():
        if name == "Metadata":
            continue
        if name not in function_map:
            print(f"WARNING: Unknown sheet name {name}")
            continue
        print(f"Processing worksheet {name}")
        data["records"].extend(function_map[name](df))
    return data


def main(source):
    inpath = pathlib.Path("transcript")/source
    outpath = pathlib.Path("json")/source
    outpath.mkdir(exist_ok=True, parents=True)

    for fn in sorted(inpath.glob("*.xls")):
        print(f"Processing {fn}")
        tables = process_file(source, fn)
        js = json.dumps(tables, indent=2)
        with (outpath / fn.name.replace(".xls", ".json")).open("w") as f:
            f.write(js)
        print()
