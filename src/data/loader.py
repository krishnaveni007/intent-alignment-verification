"""
Utilities for loading UserBench episode data from JSON files.
"""

import json
import os
from typing import Iterator


REQUIRED_KEYS = [
    "scenario_id",
    "tier",
    "dimensions",
    "u_star",
    "implicit_expressions",
    "best_ids",
    "correct_ids",
    "conversation_history",
]


def load_episode(filepath: str) -> dict:
    """Load a single UserBench episode JSON file.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the episode JSON file.

    Returns
    -------
    dict
        The parsed episode dict, guaranteed to contain all required keys.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If any required key is missing from the loaded JSON.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        episode = json.load(f)

    missing = [k for k in REQUIRED_KEYS if k not in episode]
    if missing:
        raise ValueError(
            f"Episode file '{filepath}' is missing required keys: {missing}"
        )

    return episode


def iter_episodes(folder: str, tier: str = None) -> Iterator[dict]:
    """Yield all valid episodes from JSON files in a folder.

    Parameters
    ----------
    folder : str
        Path to the directory containing episode JSON files.
    tier : str, optional
        If provided (``'easy'``, ``'medium'``, or ``'hard'``), only episodes
        whose ``tier`` field matches this value (case-insensitive) are yielded.

    Yields
    ------
    dict
        Parsed and validated episode dicts.

    Notes
    -----
    Files that fail to load or validate are skipped with a printed warning so
    that a single malformed file does not abort an entire experiment run.
    """
    json_files = sorted(
        entry.path
        for entry in os.scandir(folder)
        if entry.is_file() and entry.name.endswith(".json")
    )

    for path in json_files:
        try:
            episode = load_episode(path)
        except (ValueError, json.JSONDecodeError, OSError) as exc:
            print(f"[WARNING] Skipping '{path}': {exc}")
            continue

        if tier is not None and episode.get("tier", "").lower() != tier.lower():
            continue

        yield episode
