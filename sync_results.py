"""
sync_results.py — keep international match data up to date
==========================================================

Pulls the latest results.csv before you run predictions.

    conda activate poet
    python sync_results.py

Tries git pull from martj42/international_results first (sibling repo).
Falls back to downloading the CSV into data_cache/ if git is unavailable.
"""

import subprocess
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_REPO = SCRIPT_DIR.parent / "international_results"
LOCAL_RESULTS = DATA_REPO / "results.csv"
CACHE_DIR = SCRIPT_DIR / "data_cache"
CACHE_RESULTS = CACHE_DIR / "results.csv"
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
UPSTREAM_URL = "git@github.com:martj42/international_results.git"


def _run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(DATA_REPO), *args],
        capture_output=True,
        text=True,
        check=True,
    )


def sync_via_git() -> Path | None:
    if not (DATA_REPO / ".git").is_dir():
        return None

    try:
        remotes = _run_git(["remote"]).stdout.splitlines()
        if "upstream" not in remotes:
            _run_git(["remote", "add", "upstream", UPSTREAM_URL])
        else:
            _run_git(["remote", "set-url", "upstream", UPSTREAM_URL])

        _run_git(["fetch", "upstream"])
        _run_git(["checkout", "master"])
        _run_git(["merge", "--ff-only", "upstream/master"])
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or exc.stdout or "").strip()
        print(f"git sync failed: {err or exc}")
        return None

    if not LOCAL_RESULTS.exists():
        return None

    print(f"data updated via git -> {LOCAL_RESULTS}")
    return LOCAL_RESULTS


def sync_via_download() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    resp = requests.get(RESULTS_URL, timeout=120)
    resp.raise_for_status()
    CACHE_RESULTS.write_bytes(resp.content)
    print(f"data updated via download -> {CACHE_RESULTS}")
    return CACHE_RESULTS


def sync_results() -> Path:
    path = sync_via_git()
    if path is not None:
        return path
    return sync_via_download()


if __name__ == "__main__":
    sync_results()
