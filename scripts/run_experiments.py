"""
run_experiments.py — End-to-end experiment runner.

Loads UserBench episodes, runs preference inference at every turn, computes
all five evaluation metrics, and writes per-episode results to a CSV.  A
summary table grouped by tier is printed at the end.

Usage
-----
    python scripts/run_experiments.py \\
        --data_dir data/raw \\
        --output_dir data/results \\
        --tier easy \\
        --max_episodes 50
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

# Allow imports from the repo root when the script is called directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.loader import iter_episodes
from src.inference.llm_extractor import LLMExtractor
from src.evaluation.metrics import (
    task_success_rate,
    intent_error,
    misalignment_drift,
    regret,
    false_alignment_rate,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run intent alignment experiments over UserBench episodes."
    )
    parser.add_argument(
        "--data_dir",
        default="data/raw",
        help="Directory containing episode JSON files (default: data/raw).",
    )
    parser.add_argument(
        "--output_dir",
        default="data/results",
        help="Directory to write CSV results (default: data/results).",
    )
    parser.add_argument(
        "--tier",
        default="all",
        choices=["all", "easy", "medium", "hard"],
        help="Episode difficulty tier to process (default: all).",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=50,
        help="Maximum number of episodes to process (default: 50).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first_best_id(episode: dict) -> tuple[str, str]:
    """Return the first dimension key and its best_id from an episode."""
    best_ids: dict = episode.get("best_ids", {})
    if not best_ids:
        return "", ""
    dim = next(iter(best_ids))
    return dim, best_ids[dim]


def _build_answers(episode: dict) -> dict:
    """
    Placeholder: build a {dimension: chosen_id} map from an episode.

    In a real run the agent's answer would come from the LLM response;
    here we use best_ids as a stand-in so the pipeline is runnable end-to-end
    before the full inference loop is wired up.
    """
    # TODO: replace with actual agent answer extraction once LLMExtractor.infer
    # is implemented and the agent response is parsed from conversation_history.
    return {dim: ids for dim, ids in episode.get("best_ids", {}).items()}


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    tier_filter = None if args.tier == "all" else args.tier
    inferrer = LLMExtractor()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        args.output_dir, f"results_{args.tier}_{timestamp}.csv"
    )

    fieldnames = [
        "scenario_id",
        "tier",
        "turn_count",
        "task_success_rate",
        "intent_error_final",
        "regret",
        "false_alignment",
        "misalignment_drift",
    ]

    all_rows: list[dict] = []

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        episode_count = 0
        for episode in iter_episodes(args.data_dir, tier=tier_filter):
            if episode_count >= args.max_episodes:
                break

            scenario_id: str = episode["scenario_id"]
            ep_tier: str = episode["tier"]
            history: list[dict] = episode["conversation_history"]
            u_star: dict = episode.get("u_star", {})
            correct_ids: dict = episode.get("correct_ids", {})
            best_ids: dict = episode.get("best_ids", {})

            # --- Inference ---------------------------------------------------
            u_hat_trajectory = inferrer.infer_incremental(history)
            u_hat_final = u_hat_trajectory[-1] if u_hat_trajectory else {}

            # --- Metrics -----------------------------------------------------
            answers = _build_answers(episode)

            tsr = task_success_rate(answers, correct_ids)
            ie_final = intent_error(u_hat_final, u_star)
            drift = misalignment_drift(u_hat_trajectory, u_star)

            # Compute per-dimension regret and average across dimensions.
            regret_values = []
            for dim, best_id in best_ids.items():
                chosen = answers.get(dim, "")
                dim_correct = correct_ids.get(dim, [])
                regret_values.append(regret(chosen, best_id, dim_correct))
            avg_regret = sum(regret_values) / len(regret_values) if regret_values else 0.0

            # false_alignment requires a list of episode-level dicts; here we
            # treat this episode as a single-element list.
            ep_result = {
                "task_success": tsr == 1.0,
                "regret": avg_regret,
            }
            fa = false_alignment_rate([ep_result])

            row = {
                "scenario_id": scenario_id,
                "tier": ep_tier,
                "turn_count": len(history),
                "task_success_rate": round(tsr, 4),
                "intent_error_final": round(ie_final, 4),
                "regret": round(avg_regret, 4),
                "false_alignment": round(fa, 4),
                "misalignment_drift": json.dumps([round(v, 4) for v in drift]),
            }
            writer.writerow(row)
            all_rows.append(row)
            episode_count += 1
            print(f"  [{episode_count:>4}] {scenario_id} ({ep_tier}) — "
                  f"TSR={tsr:.2f}  IE={ie_final:.2f}  regret={avg_regret:.2f}")

    print(f"\nResults written to: {output_path}")
    _print_summary(all_rows)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _print_summary(rows: list[dict]) -> None:
    """Print a summary table of mean metrics grouped by tier."""
    from collections import defaultdict

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["tier"]].append(row)

    header = f"{'Tier':<10} {'N':>5} {'TSR':>8} {'IE_final':>10} {'Regret':>8} {'FalseAlign':>12}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for tier in ["easy", "medium", "hard"]:
        group = groups.get(tier, [])
        if not group:
            continue
        n = len(group)
        mean = lambda key: sum(float(r[key]) for r in group) / n  # noqa: E731
        print(
            f"{tier:<10} {n:>5} "
            f"{mean('task_success_rate'):>8.4f} "
            f"{mean('intent_error_final'):>10.4f} "
            f"{mean('regret'):>8.4f} "
            f"{mean('false_alignment'):>12.4f}"
        )

    # Totals row
    if rows:
        n = len(rows)
        mean = lambda key: sum(float(r[key]) for r in rows) / n  # noqa: E731
        print("-" * len(header))
        print(
            f"{'ALL':<10} {n:>5} "
            f"{mean('task_success_rate'):>8.4f} "
            f"{mean('intent_error_final'):>10.4f} "
            f"{mean('regret'):>8.4f} "
            f"{mean('false_alignment'):>12.4f}"
        )
    print("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    run(parse_args())
