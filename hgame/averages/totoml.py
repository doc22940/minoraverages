import pathlib

import damm
import numpy as np
import pandas as pd
import toml


def dropnull(rec):
    return {k: v.strip() if isinstance(v, str) else v
            for (k, v) in rec.items()
            if not pd.isnull(v)}


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
                                                df.loc[0]["league__season"]))
    return df


def add_row_metadata(df, table, prefix):
    df.insert(loc=0, column='_table', value=table)
    df.insert(loc=1, column='_key', value=np.arange(len(df))+1)
    df['_key'] = df['_key'].apply(
        lambda x: f"{prefix}{x:04}{damm.encode('%04d' % x)}"
    )
    return df


def extract_club_splits(df, prefix):
    def split_games(x):
        if pd.isnull(x):
            return x
        if "@" in x:
            return x.split("@")[0]
        else:
            return None

    i = 0
    if "team__1__team__name" in df:
        multiteam = ~df["team__1__team__name"].isnull()
    else:
        multiteam = ~df["team__0__team__name"].isnull()
    while True:
        colname = f"team__{i}__team__name"
        if colname not in df:
            return df
        df.insert(loc=df.columns.get_loc(colname)+1,
                  column=f"team__{i}__{prefix}_G",
                  value=df[colname].apply(split_games))
        if "totals__F_POS" in df:
            df.insert(loc=df.columns.get_loc(colname)+1,
                      column=f"team__{i}__F_POS",
                      value=df.loc[multiteam, "totals__F_POS"])
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

    for col in df:
        if col.endswith(("R_PCT", "B_AVG", "B_SLG", "P_AVG", "P_PCT",
                         "F_PCT", "F_UT_PCT")):
            df[col] = (
                df[col].replace("1", "1.000").replace("0", "0.000")
                .str.ljust(5, "0").str.lstrip("0")
            )
        elif col.endswith("P_ERA"):
            df[col] = df[col].apply(format_era)
    return df


def rename_columns(df, column_map):
    unknown = [c for c in df.columns if c not in column_map.keys()]
    if unknown:
        print(f"  WARNING: Unknown columns {unknown}")
    df = df.rename(columns=column_map)
    if "league__phase" not in df:
        df.insert(loc=df.columns.get_loc("league__name")+1,
                  column="league__phase", value="regular")
    else:
        col = df["league__phase"]
        df = df.drop(labels=["league__phase"], axis=1)
        df.insert(loc=df.columns.get_loc("league__name")+1,
                  column="league__phase", value=col)
    return df


def reorder_columns(df, person=True, record_type="playing"):
    if person:
        df = df.rename(
            mapper=lambda x: (
                f"{record_type}__0__{x}"
                if x.startswith(("totals", "team", "league"))
                else x
            ),
            axis='columns'
        )
    return df[
        [c for c in df.columns if not c.startswith(record_type)] +
        [c for c in df.columns if c.startswith(record_type)]
    ]


def extract_batting_team(df):
    column_map = {
        "year": "league__season",
        "nameLeague": "league__name",
        "nameClub": "name__short",
        "G": "totals__B_G",
        "AB": "totals__B_AB",
        "R": "totals__B_R",
        "OR": "totals__P_R",
        "H": "totals__B_H",
        "TB": "totals__B_TB",
        "H2B": "totals__B_2B",
        "H3B": "totals__B_3B",
        "HR": "totals__B_HR",
        "RBI": "totals__B_RBI",
        "BB": "totals__B_BB",
        "IBB": "totals__B_IBB",
        "SO": "totals__B_SO",
        "GDP": "totals__B_GDP",
        "HP": "totals__B_HP",
        "SH": "totals__B_SH",
        "SF": "totals__B_SF",
        "SB": "totals__B_SB",
        "CS": "totals__B_CS",
        "LOB": "totals__B_LOB",
        "AVG": "totals__B_AVG",
    }
    df = (
        df.pipe(rename_columns, column_map)
        .pipe(add_row_metadata, "team_batting", "TB")
        .pipe(format_percentages)
        .pipe(reorder_columns, person=False)
    )
    return {"team": [dropnull(x) for x in df.to_dict(orient='records')]}


def extract_pitching_team(df):
    column_map = {
        "year": "league__season",
        "nameLeague": "league__name",
        "nameClub": "name__short",
        "GP": "totals__P_G",
        "APP": "totals__P_APP",
        "CG": "totals__P_CG",
        "SHO": "totals__P_SHO",
        "IP": "totals__P_IP",
        "AB": "totals__P_AB",
        "R": "totals__P_R",
        "ER": "totals__P_ER",
        "H": "totals__P_H",
        "HR": "totals__P_HR",
        "BB": "totals__P_BB",
        "IBB": "totals__P_IBB",
        "SO": "totals__P_SO",
        "HB": "totals__P_HP",
        "SH": "totals__P_SH",
        "SF": "totals__P_SF",
        "WP": "totals__P_WP",
        "BK": "totals__P_BK",
        "CS": "totals__P_CS",
        "ERA": "totals__P_ERA",
    }
    df = (
        df.pipe(rename_columns, column_map)
        .pipe(add_row_metadata, "team_pitching", "TP")
        .pipe(format_percentages)
        .pipe(reorder_columns, person=False)
    )
    return {"team": [dropnull(x) for x in df.to_dict(orient='records')]}


def extract_fielding_team(df):
    column_map = {
        "year": "league__season",
        "nameLeague": "league__name",
        "nameClub": "name__short",
        "G": "totals__F_G",
        "TC": "totals__F_TC",
        "PO": "totals__F_PO",
        "A": "totals__F_A",
        "E": "totals__F_E",
        "DP": "totals__F_DP",
        "TP": "totals__F_TP",
        "PB": "totals__F_PB",
        "SB": "totals__F_SB",
        "CS": "totals__F_CS",
        "PCT": "totals__F_PCT",
    }
    df = (
        df.pipe(rename_columns, column_map)
        .pipe(add_row_metadata, "team_fielding", "TF")
        .pipe(format_percentages)
        .pipe(reorder_columns, person=False)
    )
    return {"team": [dropnull(x) for x in df.to_dict(orient='records')]}


def extract_standings_team(df):
    column_map = {
        "year": "league__season",
        "nameLeague": "league__name",
        "nameClub": "name__short",
        "phase": "league__phase",
        "division": "totals__S_DIVISION",
        "W": "totals__R_W",
        "L": "totals__R_L",
        "T": "totals__R_T",
        "PCT": "totals__R_PCT",
        "RANK": "totals__R_RANK",
    }
    df = (
        df.pipe(rename_columns, column_map)
        .pipe(add_row_metadata, "team_standings", "TS")
        .pipe(format_percentages)
        .pipe(reorder_columns, person=False)
    )
    return {"team": [dropnull(x) for x in df.to_dict(orient='records')]}


def extract_attendance_team(df):
    column_map = {
        "year": "league__season",
        "nameLeague": "league__name",
        "nameClub": "name__short",
        "ATT": "totals__R_ATT",
    }
    df = (
        df.pipe(rename_columns, column_map)
        .pipe(add_row_metadata, "team_attendance", "TA")
        .pipe(format_percentages)
        .pipe(reorder_columns, person=False)
    )
    return {"team": [dropnull(x) for x in df.to_dict(orient='records')]}


def extract_batting_individual(df):
    column_map = {
        "year": "league__season",
        "nameLeague": "league__name",
        "nameLast": "name__last",
        "nameFirst": "name__given",
        "nameClub": "team__0__team__name",
        "nameClub1": "team__0__team__name",
        "nameClub2": "team__1__team__name",
        "nameClub3": "team__2__team__name",
        "nameClub4": "team__3__team__name",
        "nameClub5": "team__4__team__name",
        "bats": "description__bats",
        "G": "totals__B_G",
        "AB": "totals__B_AB",
        "R": "totals__B_R",
        "H": "totals__B_H",
        "TB": "totals__B_TB",
        "H2B": "totals__B_2B",
        "H3B": "totals__B_3B",
        "HR": "totals__B_HR",
        "RBI": "totals__B_RBI",
        "BB": "totals__B_BB",
        "IBB": "totals__B_IBB",
        "SO": "totals__B_SO",
        "GDP": "totals__B_GDP",
        "HP": "totals__B_HP",
        "SH": "totals__B_SH",
        "SF": "totals__B_SF",
        "SB": "totals__B_SB",
        "CS": "totals__B_CS",
        "AVG": "totals__B_AVG",
        "AVG_RANK": "totals__B_AVG_RANK",
    }
    df = (
        df.pipe(rename_columns, column_map)
        .pipe(add_row_metadata, "person_batting", "B")
        .pipe(extract_club_splits, "B")
        .pipe(format_percentages)
        .pipe(reorder_columns)
    )
    return {"person": [dropnull(x) for x in df.to_dict(orient='records')]}


def extract_pitching_individual(df):
    column_map = {
        "year": "league__season",
        "nameLeague": "league__name",
        "nameLast": "name__last",
        "nameFirst": "name__given",
        "nameClub": "team__0__team__name",
        "nameClub1": "team__0__team__name",
        "nameClub2": "team__1__team__name",
        "nameClub3": "team__2__team__name",
        "nameClub4": "team__3__team__name",
        "nameClub5": "team__4__team__name",
        "throws": "description__throws",
        "GP": "totals__P_G",
        "GS": "totals__P_GS",
        "CG": "totals__P_CG",
        "SHO": "totals__P_SHO",
        "W": "totals__P_W",
        "L": "totals__P_L",
        "PCT": "totals__P_PCT",
        "IP": "totals__P_IP",
        "R": "totals__P_R",
        "ER": "totals__P_ER",
        "H": "totals__P_H",
        "HR": "totals__P_HR",
        "BB": "totals__P_BB",
        "IBB": "totals__P_IBB",
        "SO": "totals__P_SO",
        "HB": "totals__P_HP",
        "WP": "totals__P_WP",
        "BK": "totals__P_BK",
        "ERA": "totals__P_ERA",
        "ERA_RANK": "totals__P_ERA_RANK",
    }
    df = (
        df.pipe(rename_columns, column_map)
        .pipe(add_row_metadata, "person_pitching", "P")
        .pipe(extract_club_splits, "P")
        .pipe(format_percentages)
        .pipe(reorder_columns)
    )
    return {"person": [dropnull(x) for x in df.to_dict(orient='records')]}


def recode_fielding_columns(rec):
    pos = rec["playing__0__totals__F_POS"]
    rec = {k.replace("F_", f"F_{pos}_"): v
           for k, v in rec.items()}
    for k in rec:
        if k.endswith(f"F_{pos}_POS") and not pd.isnull(rec[k]):
            rec[k] = "1"
    return rec


def extract_fielding_individual(df):
    column_map = {
        "year": "league__season",
        "nameLeague": "league__name",
        "nameLast": "name__last",
        "nameFirst": "name__given",
        "nameClub": "team__0__team__name",
        "nameClub1": "team__0__team__name",
        "nameClub2": "team__1__team__name",
        "nameClub3": "team__2__team__name",
        "nameClub4": "team__3__team__name",
        "nameClub5": "team__4__team__name",
        "throws": "description__throws",
        "Pos": "totals__F_POS",
        "G": "totals__F_G",
        "PO": "totals__F_PO",
        "A": "totals__F_A",
        "E": "totals__F_E",
        "DP": "totals__F_DP",
        "PCT": "totals__F_PCT",
        "PB": "totals__F_PB",
        "TP": "totals__F_TP",
    }
    df = (
        df.pipe(rename_columns, column_map)
        .pipe(add_row_metadata, "person_fielding", "F")
        .pipe(extract_club_splits, "F")
        .pipe(format_percentages)
        .pipe(reorder_columns)
    )
    return {"person": [dropnull(recode_fielding_columns(x))
                       for x in df.to_dict(orient='records')]}


def extract_managing_individual(df):
    column_map = {
        "year": "league__season",
        "nameLeague": "league__name",
        "nameLast": "name__last",
        "nameFirst": "name__given",
        "nameClub": "team__0__team__name",
        "seq": "team__0__S_ORDER",
        "dateFirst": "team__0__S_FIRST",
    }
    df = (
        df.pipe(rename_columns, column_map)
        .pipe(add_row_metadata, "person_managing", "M")
        .pipe(extract_club_splits, "B")
        .pipe(format_percentages)
        .pipe(format_dates)
        .pipe(reorder_columns, record_type="managing")
    )
    return {"person": [dropnull(x) for x in df.to_dict(orient='records')]}


function_map = {
    "Standings": extract_standings_team,
    "Attendance": extract_attendance_team,
    "Managing": extract_managing_individual,
    "TeamBatting": extract_batting_team,
    "TeamPitching": extract_pitching_team,
    "TeamFielding": extract_fielding_team,
    "Batting": extract_batting_individual,
    "Pitching": extract_pitching_individual,
    "Fielding": extract_fielding_individual,
}


def dump(data):
    return toml.dumps(data).replace("__", ".")


def process_file(source, fn, outpath):
    text = ""
    for (name, df) in pd.read_excel(fn,
                                    dtype=str, sheet_name=None).items():
        if name in ["Metadata", "HeadToHead"]:
            continue
        try:
            print(f"  {name}")  
            result = function_map[name](df)
            text += dump(result)
        except KeyError as exc:
            print(exc)
    with (outpath/f"{fn.stem}.txt").open("w") as f:
        f.write(text)


def main(source):
    inpath = pathlib.Path("transcript")/source
    outpath = pathlib.Path("toml")/source
    outpath.mkdir(exist_ok=True, parents=True)

    for fn in sorted(inpath.glob("*.xls")):
        print(f"Processing {fn}")
        process_file(source, fn, outpath)
        print()
    
