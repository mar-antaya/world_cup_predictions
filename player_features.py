"""
player_features.py — squad-based features from international goal scorers
=========================================================================

Uses squads_2026.csv when available, otherwise falls back to recent
international scorers as a pseudo-squad. Features are computed from
goalscorers.csv in the sibling international_results repo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "international_results"
GOALSCORERS_PATH = DATA_DIR / "goalscorers.csv"
SQUADS_PATH = SCRIPT_DIR / "data_cache" / "squads_2026.csv"
CLUB_STATS_PATH = SCRIPT_DIR / "data_cache" / "player_club_stats.csv"

CLUB_STAT_COLUMNS = [
    "minutes_90d",
    "goals_90d",
    "assists_90d",
    "xg_90d",
    "shots_90d",
]

STARTER_WEIGHT = 1.0
BENCH_WEIGHT = 0.4

# Keep in sync with predict_today.py fixture -> results name mapping.
FIXTURE_TO_RESULTS = {
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Türkiye": "Turkey",
    "Congo DR": "DR Congo",
    "Côte d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic",
    "Curaçao": "Curacao",
    "USA": "United States",
    "Cape Verde": "Cabo Verde",
}

RESULTS_ALIASES = {
    "USA": "United States",
    "Korea Republic": "South Korea",
    "Republic of Ireland": "Ireland",
    "Türkiye": "Turkey",
    "Cape Verde": "Cabo Verde",
    "Côte d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic",
    "Curaçao": "Curacao",
    "Congo DR": "DR Congo",
    "Congo": "Republic of the Congo",
}

SQUAD_FEATURES = [
    "home_squad_goals_l10",
    "away_squad_goals_l10",
    "home_squad_scorers_l10",
    "away_squad_scorers_l10",
    "home_top_striker_goals_l10",
    "away_top_striker_goals_l10",
    "squad_goals_diff_l10",
    "home_squad_xg_per90_avg",
    "away_squad_xg_per90_avg",
    "home_starter_minutes_90d",
    "away_starter_minutes_90d",
    "squad_xg_diff",
]

DEFAULT_SQUAD_FEATURES = {
    "home_squad_goals_l10": 2.0,
    "away_squad_goals_l10": 2.0,
    "home_squad_scorers_l10": 2.0,
    "away_squad_scorers_l10": 2.0,
    "home_top_striker_goals_l10": 1.0,
    "away_top_striker_goals_l10": 1.0,
    "squad_goals_diff_l10": 0.0,
    "home_squad_xg_per90_avg": 0.25,
    "away_squad_xg_per90_avg": 0.25,
    "home_starter_minutes_90d": 800.0,
    "away_starter_minutes_90d": 800.0,
    "squad_xg_diff": 0.0,
}

LOOKBACK_DAYS = 730
SQUAD_SIZE = 15
MATCH_WINDOW = 10


def normalize_team(name: str) -> str:
    if not isinstance(name, str):
        return name
    name = FIXTURE_TO_RESULTS.get(name, name)
    return RESULTS_ALIASES.get(name, name)


def load_goalscorers(path: Path | None = None) -> pd.DataFrame:
    path = path or GOALSCORERS_PATH
    g = pd.read_csv(path)
    g["date"] = pd.to_datetime(g["date"])
    for col in ("team", "home_team", "away_team"):
        g[col] = g[col].map(normalize_team)
    g["own_goal"] = g["own_goal"].astype(str).str.upper().eq("TRUE")
    return g[~g["own_goal"]].copy()


def load_squads(path: Path | None = None) -> pd.DataFrame:
    path = path or SQUADS_PATH
    if not path.exists():
        return pd.DataFrame(columns=["team", "player_name", "position", "club", "is_starter"])
    squads = pd.read_csv(path)
    squads["team"] = squads["team"].map(normalize_team)
    if "is_starter" not in squads.columns:
        squads["is_starter"] = 0
    return squads


def load_club_stats(path: Path | None = None) -> pd.DataFrame:
    path = path or CLUB_STATS_PATH
    cols = ["player_name", "team", "club", *CLUB_STAT_COLUMNS]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    stats = pd.read_csv(path)
    stats["team"] = stats["team"].map(normalize_team)
    for col in CLUB_STAT_COLUMNS:
        if col not in stats.columns:
            stats[col] = 0.0
        stats[col] = pd.to_numeric(stats[col], errors="coerce").fillna(0.0)
    return stats


def pseudo_squad_players(
    goalscorers: pd.DataFrame,
    team: str,
    asof: pd.Timestamp,
    lookback_days: int = LOOKBACK_DAYS,
    squad_size: int = SQUAD_SIZE,
) -> set[str]:
    start = asof - pd.Timedelta(days=lookback_days)
    sub = goalscorers[
        (goalscorers["team"] == team)
        & (goalscorers["date"] >= start)
        & (goalscorers["date"] < asof)
    ]
    if sub.empty:
        return set()
    counts = sub.groupby("scorer").size().sort_values(ascending=False)
    return set(counts.head(squad_size).index)


def squad_players_for_team(
    squads: pd.DataFrame,
    goalscorers: pd.DataFrame,
    team: str,
    asof: pd.Timestamp,
) -> set[str]:
    team = normalize_team(team)
    named = squads[squads["team"] == team]
    if not named.empty:
        return set(named["player_name"].dropna().astype(str))
    return pseudo_squad_players(goalscorers, team, asof)


def last_team_matches(
    results: pd.DataFrame,
    team: str,
    asof: pd.Timestamp,
    n_matches: int = MATCH_WINDOW,
) -> pd.DataFrame:
    team = normalize_team(team)
    mask = (results["date"] < asof) & (
        (results["home_team"] == team) | (results["away_team"] == team)
    )
    return results.loc[mask].sort_values("date").tail(n_matches)


def team_goals_in_matches(results: pd.DataFrame, team: str, matches: pd.DataFrame) -> int:
    if matches.empty:
        return 0
    team = normalize_team(team)
    home_goals = matches.loc[matches["home_team"] == team, "home_score"].sum()
    away_goals = matches.loc[matches["away_team"] == team, "away_score"].sum()
    return int(home_goals + away_goals)


def squad_features_as_of(
    results: pd.DataFrame,
    goalscorers: pd.DataFrame,
    squads: pd.DataFrame,
    team: str,
    asof: pd.Timestamp,
    n_matches: int = MATCH_WINDOW,
) -> dict[str, float]:
    team = normalize_team(team)
    asof = pd.Timestamp(asof)
    players = squad_players_for_team(squads, goalscorers, team, asof)
    matches = last_team_matches(results, team, asof, n_matches)

    if matches.empty or not players:
        return {
            "squad_goals_l10": 0.0,
            "squad_scorers_l10": 0.0,
            "top_striker_goals_l10": 0.0,
            "squad_goal_share_l10": 0.0,
        }

    match_keys = set(zip(matches["date"], matches["home_team"], matches["away_team"]))
    goals = goalscorers[
        (goalscorers["team"] == team)
        & (goalscorers["date"] < asof)
        & goalscorers["scorer"].isin(players)
    ].copy()
    goals["match_key"] = list(zip(goals["date"], goals["home_team"], goals["away_team"]))
    goals = goals[goals["match_key"].isin(match_keys)]

    squad_goals = len(goals)
    squad_scorers = goals["scorer"].nunique()
    top_striker_goals = float(goals.groupby("scorer").size().max()) if squad_goals else 0.0
    team_goals = team_goals_in_matches(results, team, matches)
    squad_goal_share = squad_goals / team_goals if team_goals > 0 else 0.0

    return {
        "squad_goals_l10": float(squad_goals),
        "squad_scorers_l10": float(squad_scorers),
        "top_striker_goals_l10": top_striker_goals,
        "squad_goal_share_l10": float(squad_goal_share),
    }


def club_features_as_of(
    squads: pd.DataFrame,
    club_stats: pd.DataFrame,
    team: str,
) -> dict[str, float]:
    team = normalize_team(team)
    squad = squads[squads["team"] == team].copy()
    if squad.empty:
        return {
            "squad_xg_per90_avg": 0.0,
            "starter_minutes_90d": 0.0,
            "squad_goals_90d": 0.0,
        }

    merged = squad.merge(club_stats, on=["player_name", "team"], how="left", suffixes=("", "_stats"))
    for col in CLUB_STAT_COLUMNS:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    if "club_stats" in merged.columns:
        merged["club"] = merged["club"].fillna(merged["club_stats"])
    merged["is_starter"] = pd.to_numeric(merged["is_starter"], errors="coerce").fillna(0).astype(int)
    weights = np.where(merged["is_starter"] == 1, STARTER_WEIGHT, BENCH_WEIGHT)

    minutes = merged["minutes_90d"].to_numpy(dtype=float)
    xg = merged["xg_90d"].to_numpy(dtype=float)
    per90 = np.where(minutes > 0, xg / np.maximum(minutes / 90.0, 0.1), 0.0)

    if weights.sum() > 0:
        squad_xg_per90_avg = float(np.average(per90, weights=weights))
    else:
        squad_xg_per90_avg = 0.0

    starter_mask = merged["is_starter"] == 1
    starter_minutes_90d = float(merged.loc[starter_mask, "minutes_90d"].sum())

    return {
        "squad_xg_per90_avg": squad_xg_per90_avg,
        "starter_minutes_90d": starter_minutes_90d,
        "squad_goals_90d": float(merged["goals_90d"].sum()),
    }


def squad_match_features(
    results: pd.DataFrame,
    goalscorers: pd.DataFrame,
    squads: pd.DataFrame,
    home: str,
    away: str,
    asof: pd.Timestamp,
    club_stats: pd.DataFrame | None = None,
) -> dict[str, float]:
    hf = squad_features_as_of(results, goalscorers, squads, home, asof)
    af = squad_features_as_of(results, goalscorers, squads, away, asof)
    features = {
        "home_squad_goals_l10": hf["squad_goals_l10"],
        "away_squad_goals_l10": af["squad_goals_l10"],
        "home_squad_scorers_l10": hf["squad_scorers_l10"],
        "away_squad_scorers_l10": af["squad_scorers_l10"],
        "home_top_striker_goals_l10": hf["top_striker_goals_l10"],
        "away_top_striker_goals_l10": af["top_striker_goals_l10"],
        "squad_goals_diff_l10": hf["squad_goals_l10"] - af["squad_goals_l10"],
    }

    club_stats = club_stats if club_stats is not None else load_club_stats()
    hcf = club_features_as_of(squads, club_stats, home)
    acf = club_features_as_of(squads, club_stats, away)
    features.update(
        {
            "home_squad_xg_per90_avg": hcf["squad_xg_per90_avg"],
            "away_squad_xg_per90_avg": acf["squad_xg_per90_avg"],
            "home_starter_minutes_90d": hcf["starter_minutes_90d"],
            "away_starter_minutes_90d": acf["starter_minutes_90d"],
            "squad_xg_diff": hcf["squad_xg_per90_avg"] - acf["squad_xg_per90_avg"],
        }
    )
    return features


def add_default_squad_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, value in DEFAULT_SQUAD_FEATURES.items():
        out[col] = value
    return out


TRAIN_SQUAD_START = "2018-01-01"


def _zero_intl_squad_features() -> dict[str, float]:
    return {
        "squad_goals_l10": 0.0,
        "squad_scorers_l10": 0.0,
        "top_striker_goals_l10": 0.0,
    }


def build_squad_feature_lookup(
    results: pd.DataFrame,
    goalscorers: pd.DataFrame,
    start_date: str = TRAIN_SQUAD_START,
) -> dict[tuple[str, pd.Timestamp], dict[str, float]]:
    """Incremental per-team squad features (much faster than per-row recompute)."""
    results = results.sort_values("date").reset_index(drop=True)
    start = pd.Timestamp(start_date)
    lookup: dict[tuple[str, pd.Timestamp], dict[str, float]] = {}
    teams = pd.unique(pd.concat([results["home_team"], results["away_team"]]))

    for team in teams:
        team = normalize_team(team)
        team_matches = results[
            (results["home_team"] == team) | (results["away_team"] == team)
        ].sort_values("date")
        history: list[pd.Series] = []

        for _, match in team_matches.iterrows():
            asof = pd.Timestamp(match["date"])
            if asof < start:
                history.append(match)
                continue

            players = pseudo_squad_players(goalscorers, team, asof)
            prior = history[-10:]
            if not prior or not players:
                lookup[(team, asof)] = _zero_intl_squad_features()
            else:
                prior_keys = {
                    (pd.Timestamp(m["date"]), m["home_team"], m["away_team"]) for m in prior
                }
                goals = goalscorers[
                    (goalscorers["team"] == team)
                    & (goalscorers["date"] < asof)
                    & (goalscorers["scorer"].isin(players))
                ].copy()
                goals["match_key"] = list(
                    zip(goals["date"], goals["home_team"], goals["away_team"])
                )
                goals = goals[goals["match_key"].isin(prior_keys)]
                squad_goals = len(goals)
                lookup[(team, asof)] = {
                    "squad_goals_l10": float(squad_goals),
                    "squad_scorers_l10": float(goals["scorer"].nunique()),
                    "top_striker_goals_l10": float(
                        goals.groupby("scorer").size().max() if squad_goals else 0.0
                    ),
                }
            history.append(match)

    return lookup


def attach_squad_lookup(
    r: pd.DataFrame,
    lookup: dict[tuple[str, pd.Timestamp], dict[str, float]],
) -> pd.DataFrame:
    out = add_default_squad_features(r)
    home_goals, home_scorers, home_top = [], [], []
    away_goals, away_scorers, away_top = [], [], []
    for _, row in out.iterrows():
        asof = pd.Timestamp(row["date"])
        hf = lookup.get((row["home_team"], asof), _zero_intl_squad_features())
        af = lookup.get((row["away_team"], asof), _zero_intl_squad_features())
        home_goals.append(hf["squad_goals_l10"])
        home_scorers.append(hf["squad_scorers_l10"])
        home_top.append(hf["top_striker_goals_l10"])
        away_goals.append(af["squad_goals_l10"])
        away_scorers.append(af["squad_scorers_l10"])
        away_top.append(af["top_striker_goals_l10"])
    out["home_squad_goals_l10"] = home_goals
    out["home_squad_scorers_l10"] = home_scorers
    out["home_top_striker_goals_l10"] = home_top
    out["away_squad_goals_l10"] = away_goals
    out["away_squad_scorers_l10"] = away_scorers
    out["away_top_striker_goals_l10"] = away_top
    out["squad_goals_diff_l10"] = out["home_squad_goals_l10"] - out["away_squad_goals_l10"]
    return out


def add_historical_squad_features(
    r: pd.DataFrame,
    goalscorers: pd.DataFrame,
    squads: pd.DataFrame | None = None,
    use_pseudo_squads: bool = True,
) -> pd.DataFrame:
    """Attach trainable intl squad features; club columns stay at defaults."""
    del squads, use_pseudo_squads
    lookup = build_squad_feature_lookup(r, goalscorers)
    return attach_squad_lookup(r, lookup)
