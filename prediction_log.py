"""
prediction_log.py — log predictions and score them against real results
=======================================================================

Every predict_today.py run appends one row. After matches are played,
sync_results.py updates international_results and score_predictions.py
fills in outcomes + accuracy metrics.

    python score_predictions.py
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_PATH = SCRIPT_DIR / "predictions" / "prediction_log.csv"

LOG_COLUMNS = [
    "predicted_at",
    "match_date",
    "home_team",
    "away_team",
    "group_name",
    "stadium",
    "p_home",
    "p_draw",
    "p_away",
    "pick",
    "pick_confidence",
    "tag",
    "home_elo",
    "away_elo",
    "pred_home_goals",
    "pred_away_goals",
    "exp_home_goals",
    "exp_away_goals",
    "actual_home_score",
    "actual_away_score",
    "actual_result",
    "correct_pick",
    "log_loss",
    "scored_at",
]


def _empty_log() -> pd.DataFrame:
    return pd.DataFrame(columns=LOG_COLUMNS)


def load_log() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return _empty_log()
    log = pd.read_csv(LOG_PATH)
    for col in LOG_COLUMNS:
        if col not in log.columns:
            log[col] = np.nan
    return log[LOG_COLUMNS]


def log_prediction(
    match: dict,
    p_home: float,
    p_draw: float,
    p_away: float,
    pick: str,
    pick_confidence: float,
    tag: str,
    home_elo: float,
    away_elo: float,
    pred_home_goals: float | int | None = None,
    pred_away_goals: float | int | None = None,
    exp_home_goals: float | None = None,
    exp_away_goals: float | None = None,
) -> Path:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = load_log()
    key = (str(match["date"]), match["home_disp"], match["away_disp"])
    if not log.empty:
        dup = (
            (log["match_date"].astype(str) == key[0])
            & (log["home_team"] == key[1])
            & (log["away_team"] == key[2])
            & log["actual_result"].isna()
        )
        log = log.loc[~dup].copy()

    row = {
        "predicted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "match_date": match["date"],
        "home_team": match["home_disp"],
        "away_team": match["away_disp"],
        "group_name": match.get("group", ""),
        "stadium": match.get("stadium", ""),
        "p_home": round(p_home, 4),
        "p_draw": round(p_draw, 4),
        "p_away": round(p_away, 4),
        "pick": pick,
        "pick_confidence": round(pick_confidence, 4),
        "tag": tag,
        "home_elo": round(home_elo, 1),
        "away_elo": round(away_elo, 1),
        "pred_home_goals": pred_home_goals,
        "pred_away_goals": pred_away_goals,
        "exp_home_goals": round(exp_home_goals, 2) if exp_home_goals is not None else np.nan,
        "exp_away_goals": round(exp_away_goals, 2) if exp_away_goals is not None else np.nan,
        "actual_home_score": np.nan,
        "actual_away_score": np.nan,
        "actual_result": np.nan,
        "correct_pick": np.nan,
        "log_loss": np.nan,
        "scored_at": np.nan,
    }
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    log.to_csv(LOG_PATH, index=False)
    return LOG_PATH


def _normalize_log_team(name: str) -> str:
    from player_features import normalize_team

    return normalize_team(name)


def _find_result(results: pd.DataFrame, home: str, away: str, match_date: pd.Timestamp):
    home_n = _normalize_log_team(home)
    away_n = _normalize_log_team(away)
    day = pd.Timestamp(match_date).normalize()
    day_matches = results[results["date"].dt.normalize() == day]
    exact = day_matches[
        (day_matches["home_team"] == home_n) & (day_matches["away_team"] == away_n)
    ]
    exact = exact.dropna(subset=["home_score", "away_score"])
    if not exact.empty:
        row = exact.iloc[-1]
        return int(row["home_score"]), int(row["away_score"])
    reverse = day_matches[
        (day_matches["home_team"] == away_n) & (day_matches["away_team"] == home_n)
    ]
    reverse = reverse.dropna(subset=["home_score", "away_score"])
    if not reverse.empty:
        row = reverse.iloc[-1]
        return int(row["away_score"]), int(row["home_score"])
    return None


def score_predictions(results: pd.DataFrame | None = None) -> pd.DataFrame:
    log = load_log()
    if log.empty:
        print(f"no predictions logged yet -> {LOG_PATH}")
        return log

    if results is None:
        from predict_today import fetch_results, normalize_country

        results = fetch_results(sync=False)
        results["home_team"] = results["home_team"].map(normalize_country)
        results["away_team"] = results["away_team"].map(normalize_country)
        results["date"] = pd.to_datetime(results["date"])

    scored_rows = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for idx, row in log.iterrows():
        if pd.notna(row.get("actual_result")):
            scored_rows.append(row)
            continue
        found = _find_result(results, row["home_team"], row["away_team"], row["match_date"])
        if found is None:
            scored_rows.append(row)
            continue
        home_score, away_score = found
        if home_score > away_score:
            actual = row["home_team"]
            label = 0
        elif home_score == away_score:
            actual = "Draw"
            label = 1
        else:
            actual = row["away_team"]
            label = 2

        proba = np.array([[row["p_home"], row["p_draw"], row["p_away"]]])
        from sklearn.metrics import log_loss

        ll = float(log_loss([label], proba, labels=[0, 1, 2]))
        updated = row.copy()
        updated["actual_home_score"] = home_score
        updated["actual_away_score"] = away_score
        updated["actual_result"] = actual
        updated["correct_pick"] = int(row["pick"] == actual)
        updated["log_loss"] = ll
        updated["scored_at"] = now
        scored_rows.append(updated)

    out = pd.DataFrame(scored_rows)
    out.to_csv(LOG_PATH, index=False)

    finished = out[out["actual_result"].notna()].copy()
    pending = out[out["actual_result"].isna()]
    print(f"log -> {LOG_PATH}")
    print(f"scored: {len(finished)}   pending: {len(pending)}")
    if not finished.empty:
        print(f"pick accuracy     : {finished['correct_pick'].mean():.3f}")
        print(f"mean log-loss     : {finished['log_loss'].mean():.3f}")
        print(f"avg pick confidence: {finished['pick_confidence'].mean():.3f}")
    return out


if __name__ == "__main__":
    score_predictions()
