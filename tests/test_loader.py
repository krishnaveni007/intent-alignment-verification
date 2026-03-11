"""
Tests for src/data/loader.py
"""

import json
import os
import tempfile

import pytest

from src.data.loader import load_episode, iter_episodes


VALID_EPISODE = {
    "scenario_id": "ep_001",
    "tier": "easy",
    "dimensions": ["rental_car"],
    "u_star": {"rental_car": ["prefer liability insurance"]},
    "implicit_expressions": ["I have two kids"],
    "best_ids": {"rental_car": "car_A"},
    "correct_ids": {"rental_car": ["car_A", "car_B"]},
    "conversation_history": [
        {"role": "user", "content": "I need to rent a car."},
        {"role": "assistant", "content": "Sure, what are your preferences?"},
    ],
}


def _write_json(directory: str, filename: str, data: dict) -> str:
    path = os.path.join(directory, filename)
    with open(path, "w") as f:
        json.dump(data, f)
    return path


class TestLoadEpisode:
    def test_load_valid_episode(self, tmp_path):
        path = _write_json(str(tmp_path), "ep_001.json", VALID_EPISODE)
        episode = load_episode(path)
        assert episode["scenario_id"] == "ep_001"
        assert episode["tier"] == "easy"

    def test_missing_required_key_raises(self, tmp_path):
        bad = {k: v for k, v in VALID_EPISODE.items() if k != "u_star"}
        path = _write_json(str(tmp_path), "bad.json", bad)
        with pytest.raises(ValueError, match="u_star"):
            load_episode(path)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_episode("/nonexistent/path/ep.json")

    def test_invalid_json_raises(self, tmp_path):
        path = os.path.join(str(tmp_path), "bad.json")
        with open(path, "w") as f:
            f.write("not valid json {{{")
        with pytest.raises(Exception):
            load_episode(path)


class TestIterEpisodes:
    def test_yields_all_episodes(self, tmp_path):
        for i in range(3):
            ep = {**VALID_EPISODE, "scenario_id": f"ep_{i:03d}"}
            _write_json(str(tmp_path), f"ep_{i:03d}.json", ep)
        results = list(iter_episodes(str(tmp_path)))
        assert len(results) == 3

    def test_tier_filter(self, tmp_path):
        easy = {**VALID_EPISODE, "scenario_id": "easy_001", "tier": "easy"}
        hard = {**VALID_EPISODE, "scenario_id": "hard_001", "tier": "hard"}
        _write_json(str(tmp_path), "easy.json", easy)
        _write_json(str(tmp_path), "hard.json", hard)

        easy_results = list(iter_episodes(str(tmp_path), tier="easy"))
        assert len(easy_results) == 1
        assert easy_results[0]["tier"] == "easy"

    def test_skips_malformed_files(self, tmp_path, capsys):
        _write_json(str(tmp_path), "valid.json", VALID_EPISODE)
        bad_path = os.path.join(str(tmp_path), "bad.json")
        with open(bad_path, "w") as f:
            f.write("not json")

        results = list(iter_episodes(str(tmp_path)))
        assert len(results) == 1
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_empty_folder(self, tmp_path):
        results = list(iter_episodes(str(tmp_path)))
        assert results == []
