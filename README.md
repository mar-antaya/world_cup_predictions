# World Cup match predictor

A small, readable model that predicts the outcome of a single 2026 World Cup match — win / draw / loss probabilities, plus a chart you can actually post. You give it two teams, it does the rest.

I built this as a "first real ML project" you can fork and learn from. The whole thing is one file, no notebooks, no framework soup. If you've ever wanted to build a sports prediction model and didn't know where to start, start here.

```
python predict_today.py "Saudi Arabia" "Uruguay"
```

```
============================================================
  Saudi Arabia vs Uruguay
  2026-06-15  ·  Group H  ·  Miami Stadium
============================================================
  Saudi Arabia           win     9.2%
  Draw                          17.6%
  Uruguay                win    73.2%
------------------------------------------------------------
  PICK: Uruguay  (73.2%)   [LOCK]
============================================================
```

## How it works

There's no magic here — most of the work is in the features, not the model. For any match it builds:

- **Elo ratings.** Computed from scratch over every international result since 2006. Each team starts at 1500 and trades points after every game, with a bigger swing for blowouts and upsets. This one number carries most of the signal.
- **Recent form.** Win rate and goal difference over each team's last 5 and 10 matches.
- **Rest days.** How long since each team last played.
- **Head-to-head.** How these two specific teams have done against each other historically.
- **Context flags.** Neutral venue, and how meaningful the match was (a World Cup game counts for more than a friendly).

Those features go into an **XGBoost** classifier that outputs three probabilities. I went with gradient-boosted trees because this is tabular data with a few thousand rows — that's exactly where XGBoost beats both linear models (it picks up interactions on its own) and neural nets (which want far more data). It also trains in seconds and tells you which features it leaned on.

The model is trained only on matches *before* the one you're predicting, so it can't peek at the future. On a held-out validation set it lands around **60% accuracy** with a log-loss of **0.86 vs. 1.05** for a no-skill baseline — a real edge, but not a crystal ball (more on that below).

The fixtures, dates, groups and stadiums are read from `data_cache/fixtures.csv`, so you only ever type team names.

## Setup

```
pip install -r requirements.txt
python predict_today.py "Spain" "Cabo Verde"
```

Team order doesn't matter, and common spellings work (typing `Iran` is fine even though the schedule lists `IR Iran`). Run it with no arguments and it'll ask you for the two teams. The historical results file (~5MB) downloads automatically the first time you run it.

Each run also drops a branded probability chart in `predictions/<date>/`.

## What it doesn't do (yet)

Being honest about this matters more than the accuracy number. The model rates *teams*, not the eleven players actually on the pitch, so:

- No injuries or suspensions.
- No expected goals (xG) — which is the stat that actually moves modern soccer models.
- No lineups, no manager/tactics, no "this team only needs a draw to advance" context.

It also under-calls draws, like most win/draw/loss models do, because draws don't have a clean statistical fingerprint. Soccer is low-scoring and high-variance, so even a good model gets plenty wrong — judge it over a season of games with log-loss, not on any single result.

Next on my list: pulling in xG and injury/lineup data, and trying a goals-based (Poisson) approach that handles draws more naturally.

## Data

Historical results come from the open [martj42/international_results](https://github.com/martj42/international_results) dataset. Fixtures are the official 2026 schedule.

## License

MIT — do whatever you want with it. If you build something cool on top, I'd love to see it and make sure you tag me @mar_antaya on Tiktok, Youtube and Instagram or Mariana Antaya on Linkedin!
