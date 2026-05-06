"""
identify_s.py

For each (episode, dimension) pair in the existing intent_scores CSV,
identifies the task-relevant preference subset S by asking the LLM judge
which preference(s) each WRONG option violates.

Logic:
- correct options satisfy ALL preferences in u* by UserBench definition
- wrong options violate AT LEAST ONE preference
- If wrong option X violates preference i → preference i is needed to
  filter out X → i ∈ S
- S = union of violated preferences across all wrong options in database

This is cheaper than profiling all options because:
- We only call judge on wrong options (typically ~10 per dimension)
- One judge call per dimension (batch all wrong options together)

Output: adds S-related columns to a new CSV.

Usage:
    python identify_s.py --input data/results/intent_scores.csv
                         --raw   data/raw
                         --output data/results/intent_scores_with_s.csv
"""

import os
import json
import csv
import re
import time
import argparse
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from collections import Counter

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

MODEL     = "gpt-4o-mini"
SLEEP_SEC = 0.5

INITIAL_TO_DIM = {
    "F": "flight",
    "A": "apartment",
    "C": "rental_car",
    "H": "hotel",
    "R": "restaurant",
}

# New columns added by this script
NEW_FIELDS = [
    "S",                      # pipe-separated decision-relevant preference labels
    "S_size",                 # |S|
    "K_redundant",            # K - |S|: preferences in u* but not needed
    "redundant_preferences",  # pipe-separated preferences in u* but NOT in S
    "s_coverage",             # fraction of S that agent inferred (û_possible ∩ S / |S|)
    "s_identification",       # full / partial / none / unknown
    "wrong_options_profiled", # how many wrong options were used to identify S
]


# ---------------------------------------------------------------------------
# Step 1: Extract options from conversation history
# ---------------------------------------------------------------------------

def extract_options_for_dim(conversation_history: list, dimension: str) -> list[dict]:
    """
    Parses [DATABASE] messages in conversation_history to extract all options
    for the given dimension. Returns list of option dicts with full attributes.
    
    Database messages contain JSON option objects separated by newlines.
    We look for the first database response that mentions this dimension.
    """
    dim_keywords = {
        "hotel":      ["hotel"],
        "flight":     ["flight"],
        "apartment":  ["apartment"],
        "rental_car": ["rental_car", "rental car"],
        "restaurant": ["restaurant"],
    }
    keywords = dim_keywords.get(dimension, [dimension])

    options = []
    for msg in conversation_history:
        if msg["role"] != "database":
            continue
        content = msg["content"]

        # Check if this database message is for our dimension
        content_lower = content.lower()
        if not any(kw in content_lower for kw in keywords):
            continue

        # Extract JSON objects — each option is a JSON dict on its own line
        # Pattern: lines starting with { and ending with }
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    opt = json.loads(line)
                    if "id" in opt:
                        options.append(opt)
                except json.JSONDecodeError:
                    continue

        # Return after first relevant database message
        if options:
            return options

    return options


def get_wrong_options(options: list, correct_ids: list) -> list[dict]:
    """
    Returns options that are NOT in correct_ids and are not noise.
    Noise detection: options in correct_ids are correct by definition.
    We exclude options with absurd costs OR absurd flight times.
    UserBench noise options typically have cost > 50000 or nonsensical attributes.
    """
    correct_set = set(correct_ids)
    wrong = []
    for opt in options:
        if opt["id"] in correct_set:
            continue
        # Cost-based noise filter
        cost = opt.get("cost", 0)
        if isinstance(cost, list):
            cost = min((c for c in cost if c is not None), default=0)
        if cost is None:
            cost = 0
        if cost > 50000:
            continue
        # Flight-specific: skip if layover time > 24hrs (noise indicator)
        times = opt.get("time", [])
        if isinstance(times, list) and len(times) > 1:
            layover_times = times[1::2]  # odd indices = layovers
            if any(t > 24 for t in layover_times):
                continue
        wrong.append(opt)
    return wrong


# ---------------------------------------------------------------------------
# Step 2: LLM judge to identify violated preferences per wrong option
# ---------------------------------------------------------------------------

def build_s_judge_prompt(
    wrong_options: list,
    preferences: list,
    dimension: str,
) -> str:
    """
    Asks the judge: for each wrong option, which preference(s) in u* does it violate?
    All wrong options are batched into one call.
    """
    pref_list = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(preferences))

    options_str_lines = []
    for opt in wrong_options:
        options_str_lines.append(f"Option ID: {opt['id']}")
        # Pretty print relevant fields only
        relevant = {k: v for k, v in opt.items()
                    if k not in ("id",) and v is not None}
        options_str_lines.append(json.dumps(relevant, indent=2))
        options_str_lines.append("")
    options_str = "\n".join(options_str_lines)

    prompt = f"""You are analyzing travel options to identify which user preferences each option violates.

DIMENSION: {dimension}

USER PREFERENCES (u*) — these define what a correct option must satisfy:
{pref_list}

WRONG OPTIONS — these options are known to violate at least one preference above.
For each wrong option, identify WHICH preferences it violates.

WRONG OPTIONS:
{options_str}

Rules:
- A preference is violated if the option clearly does NOT satisfy it based on its attributes.
- A preference is satisfied if the option meets it, or if there is insufficient information to determine otherwise (give benefit of doubt).
- Focus only on the preferences listed above — do not invent new ones.
- Some options may violate multiple preferences.

Respond in this exact JSON format and nothing else:
{{
  "options": [
    {{
      "option_id": "<id>",
      "violated_preferences": ["<preference label>", ...],
      "reasoning": "<one sentence>"
    }},
    ...
  ]
}}"""
    return prompt


def call_judge(prompt: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Step 3: Compute S and derived metrics
# ---------------------------------------------------------------------------

def compute_s(judge_out: dict, preferences: list) -> tuple[list, int]:
    """
    S = union of all violated preferences across wrong options.
    Returns (S_labels, n_wrong_profiled).
    """
    s_set = set()
    options_data = judge_out.get("options", [])
    for opt in options_data:
        for violated in opt.get("violated_preferences", []):
            # Match to closest preference label in u*
            # (judge may paraphrase slightly)
            for pref in preferences:
                if violated.lower().strip() == pref.lower().strip():
                    s_set.add(pref)
                    break
            else:
                # Fuzzy match — if violated string is substring of any pref
                for pref in preferences:
                    if violated.lower() in pref.lower() or pref.lower() in violated.lower():
                        s_set.add(pref)
                        break

    return list(s_set), len(options_data)


def compute_s_coverage(
    s_labels: list,
    u_hat_possible_str: str,
) -> float:
    """
    Fraction of S that the agent inferred from H_t.
    s_coverage = |û_possible ∩ S| / |S|
    """
    if not s_labels:
        return 1.0  # vacuously true if S is empty

    inferred = set(u_hat_possible_str.split(" | ")) if u_hat_possible_str else set()
    s_set    = set(s_labels)
    covered  = len(inferred & s_set)
    return round(covered / len(s_set), 4)


def assign_s_identification(
    action_score: float,
    s_coverage: float,
) -> str:
    """
    full    : agent inferred all of S → correct answer was justified
    partial : agent inferred some of S but not all → partially justified
    none    : agent inferred none of S → completely unjustified correct answer
    unknown : no answer submitted (action_score = 0, can't determine)
    """
    if action_score == 0.0:
        return "unknown"
    if s_coverage == 1.0:
        return "full"
    if s_coverage > 0.0:
        return "partial"
    return "none"


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_row(
    row: dict,
    raw_dir: Path,
) -> dict:
    """
    For one CSV row (episode, dimension), loads the episode JSON,
    extracts wrong options, calls judge, computes S and derived metrics.
    Returns updated row dict with new S columns.

    Skips judge calls for misaligned cases (action_score = 0): S is only
    needed to distinguish True Alignment from False Alignment, so profiling
    wrong options for incorrect answers is wasted API budget.
    """
    episode_file = row["episode_file"]
    tier         = row["tier"]
    dimension    = row["dimension"]
    u_star_prefs = [p.strip() for p in row["u_star"].split(" | ") if p.strip()]
    correct_ids  = []  # we'll get these from the episode JSON
    action_score = float(row["action_score"])
    u_hat_possible_str = row.get("u_hat_possible", "")

    # Fast-path: misaligned cases don't need S — taxonomy is already Misalignment
    # regardless of which preferences were decision-relevant.
    if action_score == 0.0:
        return {
            **row,
            "S":                      "",
            "S_size":                 "",
            "K_redundant":            "",
            "redundant_preferences":  "",
            "s_coverage":             "",
            "s_identification":       "n/a",   # not applicable for misaligned cases
            "wrong_options_profiled": "",
        }

    # Load episode JSON
    ep_path = raw_dir / tier / episode_file
    if not ep_path.exists():
        print(f"  [WARN] Episode file not found: {ep_path}")
        return {**row, **{f: "" for f in NEW_FIELDS}}

    with open(ep_path) as f:
        episode = json.load(f)

    # Get correct_ids from dim_results
    dim_results = episode.get("dim_results", {})
    dim_res     = dim_results.get(dimension, {})
    correct_ids = dim_res.get("correct_ids", [])

    # Extract options from conversation history
    history = episode.get("conversation_history", [])
    all_options = extract_options_for_dim(history, dimension)

    if not all_options:
        print(f"  [WARN] No options found for {episode_file}/{dimension}")
        return {**row, **{f: "" for f in NEW_FIELDS}}

    # Get wrong options only
    wrong_options = get_wrong_options(all_options, correct_ids)

    if not wrong_options:
        print(f"  [WARN] No wrong options for {episode_file}/{dimension}")
        # If no wrong options, S = u* (all preferences needed)
        s_labels = u_star_prefs
        s_size   = len(s_labels)
        s_coverage = compute_s_coverage(s_labels, u_hat_possible_str)
        s_id = assign_s_identification(action_score, s_coverage)
        return {**row,
                "S":                      " | ".join(s_labels),
                "S_size":                 s_size,
                "K_redundant":            len(u_star_prefs) - s_size,
                "s_coverage":             s_coverage,
                "s_identification":       s_id,
                "wrong_options_profiled": 0}

    # Call judge
    prompt = build_s_judge_prompt(wrong_options, u_star_prefs, dimension)
    try:
        judge_out = call_judge(prompt)
        time.sleep(SLEEP_SEC)
    except Exception as e:
        print(f"  [ERROR] Judge failed for {episode_file}/{dimension}: {e}")
        return {**row, **{f: "" for f in NEW_FIELDS}}

    s_labels, n_profiled = compute_s(judge_out, u_star_prefs)
    s_size     = len(s_labels)
    k_redundant = len(u_star_prefs) - s_size
    s_coverage = compute_s_coverage(s_labels, u_hat_possible_str)
    s_id       = assign_s_identification(action_score, s_coverage)

    print(f"  {dimension}: K={len(u_star_prefs)} |S|={s_size} "
          f"redundant={k_redundant} s_coverage={s_coverage} → {s_id}")

    return {
        **row,
        "S":                      " | ".join(s_labels),
        "S_size":                 s_size,
        "K_redundant":            k_redundant,
        "s_coverage":             s_coverage,
        "s_identification":       s_id,
        "wrong_options_profiled": n_profiled,
    }


def get_already_processed(output_path: Path) -> set:
    """Returns set of (episode_file, dimension) tuples already in output CSV."""
    if not output_path.exists():
        return set()
    with open(output_path, newline="") as f:
        reader = csv.DictReader(f)
        return {(row["episode_file"], row["dimension"]) for row in reader}


def main(input_paths: list, raw_dir: Path, output_path: Path, sample: bool = False):
    # Read and combine all input CSVs
    all_rows  = []
    in_fields = None
    for input_path in input_paths:
        with open(input_path, newline="") as f:
            reader = csv.DictReader(f)
            rows   = list(reader)
            if in_fields is None:
                in_fields = reader.fieldnames
            all_rows.extend(rows)
        print(f"Loaded {len(rows)} rows from {input_path}")

    print(f"Total rows: {len(all_rows)}")

    if sample:
        all_rows = all_rows[:10]
        print(f"[SAMPLE] Processing first 10 rows only.")

    # Output fields = input fields + new S fields
    out_fields = list(in_fields) + NEW_FIELDS

    # Resume logic
    already_done = get_already_processed(output_path)
    if already_done:
        print(f"[RESUME] Skipping {len(already_done)} already-processed rows.")

    write_header = not output_path.exists()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped   = 0

    with open(output_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=out_fields)
        if write_header:
            writer.writeheader()

    processed          = 0
    skipped            = 0
    skipped_misaligned = 0

    with open(output_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=out_fields)
        if write_header:
            writer.writeheader()

        for row in all_rows:
            key = (row["episode_file"], row["dimension"])
            if key in already_done:
                skipped += 1
                continue

            action_score = float(row.get("action_score", 0))
            if action_score == 0.0:
                # Write stub — no judge call needed for misaligned cases
                stub = {**row, "S": "", "S_size": "", "K_redundant": "",
                        "redundant_preferences": "", "s_coverage": "",
                        "s_identification": "n/a", "wrong_options_profiled": ""}
                writer.writerow(stub)
                csvfile.flush()
                skipped_misaligned += 1
                continue

            print(f"\n{row['episode_file']} / {row['dimension']}")
            updated_row = process_row(row, raw_dir)
            writer.writerow(updated_row)
            csvfile.flush()
            processed += 1

    print(f"\n✓ Done. {processed} rows processed via judge, "
          f"{skipped_misaligned} misaligned skipped (no judge call), "
          f"{skipped} already done.")
    print(f"Output: {output_path}")

    # Summary of S identification (exclude n/a rows)
    if processed > 0:
        with open(output_path, newline="") as f:
            out_rows = [r for r in csv.DictReader(f)
                        if r.get("s_identification") and r["s_identification"] != "n/a"]
        counts = Counter(r["s_identification"] for r in out_rows)
        total  = len(out_rows)
        print("\nS identification summary (successful cases only):")
        for label, count in counts.most_common():
            print(f"  {label}: {count} ({100*count/total:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, nargs="+",
        default=[
            Path("data/results/intent_scores_easy.csv"),
            Path("data/results/intent_scores_medium.csv"),
            Path("data/results/intent_scores_hard.csv"),
        ],
        help="One or more intent scores CSV files to process"
    )
    parser.add_argument("--raw",    type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/results/intent_scores_with_s.csv"))
    parser.add_argument("--sample", action="store_true",
                        help="Only process first 10 rows (for validation)")
    args = parser.parse_args()
    main(args.input, args.raw, args.output, sample=args.sample)