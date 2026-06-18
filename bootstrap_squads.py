"""
bootstrap_squads.py — build squads_2026.csv from recent international scorers
=============================================================================

Creates a starter squad list for every 2026 World Cup team using the top
international goal scorers from the last two years. Edit the CSV manually
when official squads are announced.

    conda activate poet
    python bootstrap_squads.py
"""

from pathlib import Path

import pandas as pd

from player_features import (
    LOOKBACK_DAYS,
    SQUAD_SIZE,
    SQUADS_PATH,
    load_goalscorers,
    normalize_team,
)

FIXTURES_PATH = Path(__file__).resolve().parent / "data_cache" / "fixtures.csv"


def tournament_teams() -> list[str]:
    fx = pd.read_csv(FIXTURES_PATH)
    names: set[str] = set()
    skip = ("winner", "runner", "third", "place", "group")
    for teams in fx["teams"]:
        if " v " not in str(teams):
            continue
        for side in str(teams).split(" v "):
            side = side.strip()
            if not any(word in side.lower() for word in skip):
                names.add(normalize_team(side))
    return sorted(names)


def bootstrap_squads(asof: str = "2026-06-11") -> pd.DataFrame:
    goalscorers = load_goalscorers()

    asof_ts = pd.Timestamp(asof)
    start = asof_ts - pd.Timedelta(days=LOOKBACK_DAYS)

    rows: list[dict[str, object]] = []
    for team in tournament_teams():
        sub = goalscorers[
            (goalscorers["team"] == team)
            & (goalscorers["date"] >= start)
            & (goalscorers["date"] < asof_ts)
        ]
        counts = sub.groupby("scorer").size().sort_values(ascending=False)
        if counts.empty:
            rows.append(
                {
                    "team": team,
                    "player_name": "",
                    "position": "",
                    "club": "",
                    "is_starter": 0,
                    "notes": "add players manually",
                }
            )
            continue

        for rank, (player, goals) in enumerate(counts.head(SQUAD_SIZE).items(), start=1):
            rows.append(
                {
                    "team": team,
                    "player_name": player,
                    "position": "",
                    "club": "",
                    "is_starter": int(rank <= 11),
                    "notes": f"auto: {int(goals)} intl goals since {start.date()}",
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    squads = bootstrap_squads()
    SQUADS_PATH.parent.mkdir(parents=True, exist_ok=True)
    squads.to_csv(SQUADS_PATH, index=False)
    teams = squads["team"].nunique()
    players = squads[squads["player_name"].astype(str).str.len() > 0]["player_name"].nunique()
    print(f"wrote {len(squads)} rows for {teams} teams -> {SQUADS_PATH}")
    print(f"unique players: {players}")
    print("edit data_cache/squads_2026.csv to replace auto picks with official squads")


if __name__ == "__main__":
    main()
