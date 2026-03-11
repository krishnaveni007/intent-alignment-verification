# Verifying Latent Intent: Evaluation Framework

## Overview

This project investigates whether standard task success metrics are sufficient
for evaluating AI agent alignment with user preferences in multi-turn
conversational tasks.  Using episode data from UserBench, we infer a user
preference vector û_t from conversation history at each turn and compare it
against the ground-truth preference vector u*, enabling measurement of latent
misalignment that task success alone cannot detect.  The framework tests four
hypotheses: (H1) task success rate overestimates true intent alignment; (H2)
intent error decreases over the course of a conversation; (H3) false alignment
rate increases with scenario difficulty; and (H4) agents with explicit
preference tracking achieve lower regret than agents without it.

## Repo Structure

```
intent-alignment-verification/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── data/
│   ├── raw/            # Raw episode JSONs from UserBench
│   ├── processed/      # Extracted û_t snapshots per turn (JSON)
│   └── results/        # Final metrics CSVs per experiment run
│
├── scripts/
│   ├── generate_samples.py   # Placeholder — will be populated separately
│   └── run_experiments.py    # Loops over episodes, calls inference + eval,
│                             # saves results to data/results/
│
├── src/
│   ├── data/
│   │   └── loader.py         # load_episode / iter_episodes
│   ├── inference/
│   │   ├── base.py           # PreferenceInferrer ABC
│   │   └── llm_extractor.py  # LLMExtractor (placeholder)
│   └── evaluation/
│       └── metrics.py        # All five metric functions
│
└── tests/
    ├── test_loader.py
    └── test_metrics.py
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy the environment template and fill in your API keys
cp .env.example .env
# Edit .env and set OPENAI_API_KEY and/or ANTHROPIC_API_KEY
```

## How to Run

```bash
# Run on up to 50 easy-tier episodes
python scripts/run_experiments.py --tier easy --max_episodes 50

# Run on all tiers, all episodes in data/raw
python scripts/run_experiments.py --tier all

# Custom input / output directories
python scripts/run_experiments.py \
    --data_dir path/to/raw \
    --output_dir path/to/results \
    --tier medium \
    --max_episodes 100
```

Results are written to a timestamped CSV in `data/results/` and a summary
table grouped by tier is printed to stdout.

## Metrics Description

| Metric | Description | Hypothesis |
|---|---|---|
| `task_success_rate` | Fraction of dimensions where the agent chose an option in `correct_ids`. | H1 — this metric alone overestimates alignment. |
| `intent_error` | Jaccard distance between û_t and u* averaged across dimensions; lower is better. | H2 — should decrease over conversation turns. |
| `misalignment_drift` | `intent_error` computed at every turn, returning a per-turn trajectory. | H2 — plot to visualise alignment improvement. |
| `regret` | 0 if the best option was chosen, 0.5 if an acceptable-but-suboptimal option was chosen, 1.0 otherwise. | H4 — explicit preference tracking should reduce regret. |
| `false_alignment_rate` | Fraction of successful episodes (task_success=True) that still have regret above a threshold. | H3 — should increase from easy → medium → hard. |

## Data

- **`data/raw/`** — UserBench episode JSON files.  Each file encodes one
  conversation scenario and must contain the keys: `scenario_id`, `tier`,
  `dimensions`, `u_star`, `implicit_expressions`, `best_ids`, `correct_ids`,
  and `conversation_history`.  These files are excluded from version control
  (see `.gitignore`).

- **`data/processed/`** — JSON files storing û_t snapshots produced by
  `PreferenceInferrer.infer_incremental()`, one file per episode.  Persisting
  these avoids re-running inference when only the evaluation metrics change.

- **`data/results/`** — Timestamped CSV files written by
  `scripts/run_experiments.py`, one row per episode, containing all five
  metric values plus metadata (`scenario_id`, `tier`, `turn_count`).
