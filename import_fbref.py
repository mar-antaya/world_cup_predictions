"""
import_fbref.py — bulk-import player stats from FBref CSV exports
================================================================

FBref does not always offer a download button, but you can:
  1. Open a league stats page (Standard Stats / Shooting)
  2. Select the table, copy, and paste into a spreadsheet
  3. Save as CSV into data_cache/fbref_imports/

This script maps those exports onto your World Cup squads and writes
data_cache/player_club_stats_manual.csv, then runs sync_player_stats.py.

Usage:
    conda activate poet
    python import_fbref.py data_cache/fbref_imports/la_liga.csv
    python import_fbref.py --dir data_cache/fbref_imports
    python import_fbref.py data_cache/fbref_imports/*.csv --dry-run

Notes:
    - FBref league exports are usually season-to-date, not literally 90 days.
    - Values are stored in the minutes_90d / goals_90d columns as a rolling
      club-form proxy until you import true 90-day match logs.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd

from player_features import CLUB_STAT_COLUMNS, normalize_team
from sync_player_stats import MANUAL_PATH, sync_player_stats

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMPORT_DIR = SCRIPT_DIR / "data_cache" / "fbref_imports"

# FIFA 3-letter codes used in FBref "Nation" column -> World Cup team name.
NATION_CODE_TO_TEAM: dict[str, str] = {
    "ALG": "Algeria",
    "ARG": "Argentina",
    "AUS": "Australia",
    "AUT": "Austria",
    "BEL": "Belgium",
    "BIH": "Bosnia and Herzegovina",
    "BRA": "Brazil",
    "CAN": "Canada",
    "CIV": "Côte d'Ivoire",
    "COL": "Colombia",
    "CPV": "Cabo Verde",
    "CRO": "Croatia",
    "CUW": "Curaçao",
    "CZE": "Czechia",
    "COD": "Congo DR",
    "ECU": "Ecuador",
    "EGY": "Egypt",
    "ENG": "England",
    "ESP": "Spain",
    "FRA": "France",
    "GHA": "Ghana",
    "GER": "Germany",
    "HAI": "Haiti",
    "IRN": "Iran",
    "IRQ": "Iraq",
    "JOR": "Jordan",
    "JPN": "Japan",
    "KOR": "South Korea",
    "KSA": "Saudi Arabia",
    "MAR": "Morocco",
    "MEX": "Mexico",
    "NED": "Netherlands",
    "NOR": "Norway",
    "NZL": "New Zealand",
    "PAN": "Panama",
    "PAR": "Paraguay",
    "POR": "Portugal",
    "QAT": "Qatar",
    "SCO": "Scotland",
    "SEN": "Senegal",
    "SUI": "Switzerland",
    "SWE": "Sweden",
    "TUN": "Tunisia",
    "TUR": "Turkey",
    "URU": "Uruguay",
    "USA": "United States",
    "UZB": "Uzbekistan",
}

COLUMN_HINTS: dict[str, tuple[str, ...]] = {
    "player": ("player",),
    "nation": ("nation", "nat", "country"),
    "club": ("squad", "club", "team"),
    "minutes": ("min", "minutes", "playing time: min"),
    "goals": ("gls", "goals", "performance: gls"),
    "assists": ("ast", "assists", "performance: ast"),
    "xg": ("xg", "expected: xg"),
    "shots": ("sh", "shots", "standard: sh", "shooting: sh"),
}


def normalize_player_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-zA-Z0-9\s'-]", "", text)
    return re.sub(r"\s+", " ", text.lower()).strip()


def flatten_columns(columns: pd.Index) -> list[str]:
    flat: list[str] = []
    for col in columns:
        if isinstance(col, tuple):
            parts = [str(part) for part in col if str(part) not in {"", "nan", "None", "Unnamed: 0_level_0"}]
            flat.append(": ".join(parts))
        else:
            flat.append(str(col))
    return flat


def read_fbref_csv(path: Path) -> pd.DataFrame:
    single = pd.read_csv(path, header=0)
    single.columns = flatten_columns(single.columns)
    single.columns = [re.sub(r"\s+", " ", str(col)).strip() for col in single.columns]
    if pick_column(single.columns, COLUMN_HINTS["player"]):
        return single

    multi = pd.read_csv(path, header=[0, 1])
    multi.columns = flatten_columns(multi.columns)
    multi.columns = [re.sub(r"\s+", " ", str(col)).strip() for col in multi.columns]
    return multi


def pick_column(columns: Iterable[str], hints: tuple[str, ...]) -> str | None:
    normalized = {col: re.sub(r"[^a-z0-9]", "", col.lower()) for col in columns}
    for col, norm in normalized.items():
        for hint in hints:
            if norm == re.sub(r"[^a-z0-9]", "", hint):
                return col
    for col, norm in normalized.items():
        for hint in hints:
            hint_norm = re.sub(r"[^a-z0-9]", "", hint)
            if hint_norm and hint_norm in norm:
                return col
    return None


def parse_nation_code(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    match = re.search(r"\b([A-Z]{3})\b", value.upper())
    if not match:
        return None
    return NATION_CODE_TO_TEAM.get(match.group(1))


def to_float(value: object) -> float:
    if pd.isna(value):
        return 0.0
    text = str(value).strip().replace(",", "")
    if text in {"", "-"}:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def load_squad_index() -> tuple[pd.DataFrame, dict[str, list[tuple[str, str]]]]:
    from player_features import load_squads

    squads = load_squads()
    squads = squads[squads["player_name"].astype(str).str.len() > 0].copy()
    index: dict[str, list[tuple[str, str]]] = {}
    for _, row in squads.iterrows():
        key = normalize_player_name(str(row["player_name"]))
        index.setdefault(key, []).append((str(row["player_name"]), normalize_team(str(row["team"]))))
    return squads, index


def match_squad_player(
    player_name: str,
    nation_team: str | None,
    squad_index: dict[str, list[tuple[str, str]]],
) -> tuple[str, str] | None:
    key = normalize_player_name(player_name)
    candidates = squad_index.get(key, [])
    if not candidates and " " in key:
        # Last-name fallback for accented or shortened FBref names.
        last = key.split()[-1]
        for squad_key, entries in squad_index.items():
            if squad_key.endswith(last) or last in squad_key.split():
                candidates.extend(entries)

    if not candidates:
        return None
    if nation_team:
        for canonical_name, team in candidates:
            if team == nation_team:
                return canonical_name, team
    return candidates[0]


def map_fbref_frame(df: pd.DataFrame, squad_index: dict[str, list[tuple[str, str]]]) -> pd.DataFrame:
    player_col = pick_column(df.columns, COLUMN_HINTS["player"])
    nation_col = pick_column(df.columns, COLUMN_HINTS["nation"])
    club_col = pick_column(df.columns, COLUMN_HINTS["club"])
    minutes_col = pick_column(df.columns, COLUMN_HINTS["minutes"])
    goals_col = pick_column(df.columns, COLUMN_HINTS["goals"])
    assists_col = pick_column(df.columns, COLUMN_HINTS["assists"])
    xg_col = pick_column(df.columns, COLUMN_HINTS["xg"])
    shots_col = pick_column(df.columns, COLUMN_HINTS["shots"])

    if not player_col:
        raise ValueError("Could not find a Player column in FBref CSV")

    rows: list[dict[str, object]] = []
    for _, raw in df.iterrows():
        player = str(raw.get(player_col, "")).strip()
        if not player or player.lower() in {"player", "squad total", "opponent total"}:
            continue

        nation_team = parse_nation_code(raw.get(nation_col, "")) if nation_col else None
        matched = match_squad_player(player, nation_team, squad_index)
        if matched is None:
            continue

        canonical_name, team = matched
        rows.append(
            {
                "player_name": canonical_name,
                "team": team,
                "club": str(raw.get(club_col, "")).strip() if club_col else "",
                "minutes_90d": to_float(raw.get(minutes_col)) if minutes_col else 0.0,
                "goals_90d": to_float(raw.get(goals_col)) if goals_col else 0.0,
                "assists_90d": to_float(raw.get(assists_col)) if assists_col else 0.0,
                "xg_90d": to_float(raw.get(xg_col)) if xg_col else 0.0,
                "shots_90d": to_float(raw.get(shots_col)) if shots_col else 0.0,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.drop_duplicates(["player_name", "team"], keep="last")


def merge_manual_updates(imports: pd.DataFrame) -> pd.DataFrame:
    cols = ["player_name", "team", "club", *CLUB_STAT_COLUMNS]
    if MANUAL_PATH.exists():
        manual = pd.read_csv(MANUAL_PATH)
        manual["team"] = manual["team"].map(normalize_team)
        for col in cols:
            if col not in manual.columns:
                manual[col] = 0.0 if col in CLUB_STAT_COLUMNS else ""
        manual = manual[cols]
    else:
        manual = pd.DataFrame(columns=cols)

    imports = imports[cols].copy()
    combined = pd.concat([manual, imports], ignore_index=True)
    return combined.drop_duplicates(["player_name", "team"], keep="last")


def import_fbref_files(paths: list[Path], dry_run: bool = False) -> pd.DataFrame:
    _, squad_index = load_squad_index()
    imported_parts: list[pd.DataFrame] = []

    for path in paths:
        frame = read_fbref_csv(path)
        mapped = map_fbref_frame(frame, squad_index)
        print(f"{path.name}: matched {len(mapped)} squad players")
        imported_parts.append(mapped)

    if not imported_parts:
        print("no FBref files imported")
        return pd.DataFrame()

    imports = pd.concat(imported_parts, ignore_index=True)
    imports = imports.drop_duplicates(["player_name", "team"], keep="last")
    merged = merge_manual_updates(imports)

    if dry_run:
        print(f"dry run: would write {len(merged)} rows to {MANUAL_PATH}")
        return merged

    MANUAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MANUAL_PATH, index=False)
    print(f"manual club stats saved -> {MANUAL_PATH}")
    sync_player_stats()
    return merged


def collect_paths(paths: list[str], directory: str | None) -> list[Path]:
    collected: list[Path] = []
    if directory:
        collected.extend(sorted(Path(directory).glob("*.csv")))
    for item in paths:
        path = Path(item)
        if path.is_dir():
            collected.extend(sorted(path.glob("*.csv")))
        elif path.exists():
            collected.append(path)
    # Skip example/manual outputs.
    return [p for p in collected if p.name not in {"player_club_stats_manual.csv", "player_club_stats.csv"}]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import FBref CSV exports into club player stats")
    parser.add_argument("paths", nargs="*", help="FBref CSV files to import")
    parser.add_argument("--dir", default=None, help="Import every CSV in this folder")
    parser.add_argument("--dry-run", action="store_true", help="Show matches without writing files")
    args = parser.parse_args(argv)

    paths = collect_paths(args.paths, args.dir)
    if not paths and DEFAULT_IMPORT_DIR.exists():
        paths = collect_paths([], str(DEFAULT_IMPORT_DIR))
    if not paths:
        print("No FBref CSV files found.")
        print(f"Save exports under: {DEFAULT_IMPORT_DIR}")
        return 1

    import_fbref_files(paths, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
