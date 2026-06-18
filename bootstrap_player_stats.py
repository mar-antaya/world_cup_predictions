"""
bootstrap_player_stats.py — create player_club_stats.csv from the squad list
============================================================================

Builds a club-stats template (minutes, goals, assists, xG) for every player
in squads_2026.csv. Fill values manually or run sync_player_stats.py.

    conda activate poet
    python bootstrap_player_stats.py
"""

from pathlib import Path

import pandas as pd

from player_features import CLUB_STAT_COLUMNS, CLUB_STATS_PATH, load_squads

STAT_DEFAULTS = {col: 0.0 for col in CLUB_STAT_COLUMNS}


def bootstrap_player_stats() -> pd.DataFrame:
    squads = load_squads()
    players = squads[squads["player_name"].astype(str).str.len() > 0].copy()
    rows: list[dict[str, object]] = []
    for _, row in players.iterrows():
        entry = {
            "player_name": row["player_name"],
            "team": row["team"],
            "club": row.get("club", ""),
            **STAT_DEFAULTS,
            "source": "template",
        }
        rows.append(entry)
    return pd.DataFrame(rows)


def main() -> None:
    stats = bootstrap_player_stats()
    CLUB_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(CLUB_STATS_PATH, index=False)
    print(f"wrote {len(stats)} rows -> {CLUB_STATS_PATH}")
    print("fill stats manually or run: python sync_player_stats.py")


if __name__ == "__main__":
    main()
