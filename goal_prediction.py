"""
goal_prediction.py — expected goals and most likely scoreline (Poisson)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

INTL_AVG_GOALS = 1.35
MAX_GOALS = 5
ELO_BASELINE = 1500.0
OPP_ADJ_POWER = 0.5


def _poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def team_goal_rates(
    long: pd.DataFrame,
    team: str,
    asof: pd.Timestamp,
    elo: dict[str, float] | None = None,
    n: int = 10,
) -> tuple[float, float]:
    """Opponent-strength-adjusted goals scored/conceded over last n matches.

    Scoring against weak opponents is discounted; conceding against weak
    opponents is penalized. Uses current Elo as an opponent-strength proxy.
    """
    sub = long[(long["team"] == team) & (long["date"] < pd.Timestamp(asof))].sort_values("date").tail(n)
    if sub.empty:
        return INTL_AVG_GOALS, INTL_AVG_GOALS

    if not elo:
        return float(sub["gf"].mean()), float(sub["ga"].mean())

    opp_elo = sub["opp"].map(lambda o: elo.get(o, ELO_BASELINE)).to_numpy(dtype=float)
    adj = (opp_elo / ELO_BASELINE) ** OPP_ADJ_POWER
    gf = sub["gf"].to_numpy(dtype=float) * adj          # weak opp (adj<1) -> counts less
    ga = sub["ga"].to_numpy(dtype=float) / adj          # concede vs weak opp -> counts more
    return float(gf.mean()), float(ga.mean())


def predict_goals(
    long: pd.DataFrame,
    home: str,
    away: str,
    asof: pd.Timestamp,
    neutral: bool = True,
    home_elo: float = 1500.0,
    away_elo: float = 1500.0,
    elo: dict[str, float] | None = None,
) -> dict[str, float | tuple[int, int]]:
    h_gf, h_ga = team_goal_rates(long, home, asof, elo)
    a_gf, a_ga = team_goal_rates(long, away, asof, elo)

    lam_home = (h_gf / INTL_AVG_GOALS) * (a_ga / INTL_AVG_GOALS) * INTL_AVG_GOALS
    lam_away = (a_gf / INTL_AVG_GOALS) * (h_ga / INTL_AVG_GOALS) * INTL_AVG_GOALS

    elo_adj = 10 ** ((home_elo - away_elo) / 400.0)
    lam_home *= elo_adj**0.15
    lam_away /= elo_adj**0.15
    if neutral:
        lam_home *= 0.98
        lam_away *= 0.98

    lam_home = max(0.25, min(lam_home, 3.5))
    lam_away = max(0.25, min(lam_away, 3.5))

    best_score = (0, 0)
    best_p = 0.0
    score_probs: list[tuple[int, int, float]] = []
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            p = _poisson_pmf(i, lam_home) * _poisson_pmf(j, lam_away)
            score_probs.append((i, j, p))
            if p > best_p:
                best_p, best_score = p, (i, j)

    score_probs.sort(key=lambda x: x[2], reverse=True)
    top_scores = score_probs[:3]

    over_25 = sum(p for i, j, p in score_probs if i + j >= 3)
    btts = sum(p for i, j, p in score_probs if i >= 1 and j >= 1)

    return {
        "exp_home_goals": lam_home,
        "exp_away_goals": lam_away,
        "pred_home_goals": best_score[0],
        "pred_away_goals": best_score[1],
        "scoreline_prob": best_p,
        "over_2_5_prob": over_25,
        "btts_prob": btts,
        "top_scores": top_scores,
    }
