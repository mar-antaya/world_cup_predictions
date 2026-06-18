"""
sync_player_stats.py — refresh club stats for squad players
===========================================================

Merges manual updates from data_cache/player_club_stats_manual.csv into
player_club_stats.csv and adds any new squad players with zeroed stats.

    conda activate poet
    python sync_player_stats.py
"""

from pathlib import Path

import pandas as pd

from player_features import CLUB_STAT_COLUMNS, CLUB_STATS_PATH, load_squads, normalize_team

MANUAL_PATH = Path(__file__).resolve().parent / "data_cache" / "player_club_stats_manual.csv"


def load_or_bootstrap_stats() -> pd.DataFrame:
    if CLUB_STATS_PATH.exists():
        stats = pd.read_csv(CLUB_STATS_PATH)
    else:
        from bootstrap_player_stats import bootstrap_player_stats

        stats = bootstrap_player_stats()
    stats["team"] = stats["team"].map(normalize_team)
    return stats


def upsert_rows(base: pd.DataFrame, updates: pd.DataFrame, source: str) -> pd.DataFrame:
    if updates.empty:
        return base

    updates = updates.copy()
    updates["team"] = updates["team"].map(normalize_team)
    updates = updates.dropna(subset=["player_name", "team"])
    updates["source"] = source

    cols = ["player_name", "team", "club", *CLUB_STAT_COLUMNS, "source"]
    for col in cols:
        if col not in base.columns:
            base[col] = "" if col in {"club", "source"} else 0.0
        if col not in updates.columns:
            updates[col] = "" if col in {"club", "source"} else 0.0

    base = base[cols].copy()
    updates = updates[cols].copy()
    combined = pd.concat([base, updates], ignore_index=True)
    combined = combined.drop_duplicates(["player_name", "team"], keep="last")
    return combined


def sync_player_stats() -> pd.DataFrame:
    squads = load_squads()
    stats = load_or_bootstrap_stats()

    if MANUAL_PATH.exists():
        manual = pd.read_csv(MANUAL_PATH)
        stats = upsert_rows(stats, manual, source="manual")

    squad_rows = squads[squads["player_name"].astype(str).str.len() > 0][
        ["player_name", "team", "club"]
    ].copy()
    squad_rows = squad_rows.assign(
        **{col: 0.0 for col in CLUB_STAT_COLUMNS},
        source="template",
    )
    existing = set(zip(stats["player_name"], stats["team"]))
    missing = squad_rows[
        ~squad_rows.apply(lambda row: (row["player_name"], row["team"]) in existing, axis=1)
    ]
    stats = pd.concat([stats, missing], ignore_index=True)

    CLUB_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(CLUB_STATS_PATH, index=False)

    filled = (stats[CLUB_STAT_COLUMNS].sum(axis=1) > 0).sum()
    print(f"club stats saved -> {CLUB_STATS_PATH}")
    print(f"players with club stats: {filled}/{len(stats)}")
    if filled == 0:
        print(f"add rows to {MANUAL_PATH} or edit {CLUB_STATS_PATH} directly")
    return stats


if __name__ == "__main__":
    sync_player_stats()
