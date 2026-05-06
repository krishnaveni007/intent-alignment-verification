#!/usr/bin/env python3
"""
generate_samples.py

Run TravelEnv episodes across easy / medium / hard tiers using gpt-4o-mini
as the agent.

Usage:
    export OPENAI_API_KEY="sk-..."
    python3 generate_samples.py

Outputs:
    data/raw/easy_031.json  … easy_049.json   (19 more to reach 50)
    data/raw/medium_000.json … medium_049.json
    data/raw/hard_000.json   … hard_049.json

Post-hoc filtering:
    Only episodes where at least one dimension has task success
    (has_any_task_success=True) are kept for analysis.
    All episodes are saved regardless — filtering happens downstream.

Batching:
    Adjust TIER_CONFIGS below to change START_IDX / NUM_EPISODES per tier.
    The fixed random_state=42 shuffle guarantees reproducible, non-overlapping
    row selection across runs.
"""

import os
import sys
import json
import time
import yaml
import traceback
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── guard: OPENAI_API_KEY ─────────────────────────────────────────────────────
if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY is not set.")
    print("       Please run:  export OPENAI_API_KEY='sk-...'")
    sys.exit(1)

import openai
client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
print("OpenAI client initialised OK.")

# ── load tool schema ──────────────────────────────────────────────────────────
with open("schema/interact_tool.yaml") as f:
    TOOL_SCHEMA = yaml.safe_load(f)["tool_schema"]
print(f"Tool schema loaded: {TOOL_SCHEMA['function']['name']}")

# ── agent model ───────────────────────────────────────────────────────────────
AGENT_MODEL = "gpt-5.4-mini"

# ── tier configs ──────────────────────────────────────────────────────────────
# Easy:   travel22   — 2 aspects, 2 prefs each
# Medium: travel33 + travel233 — 2-3 aspects, 2-3 prefs each (pooled)
# Hard:   travel444 ONLY — 4 aspects, 4 prefs each (travel334 excluded)

TIER_CONFIGS = [
    {
        "name":         "easy",
        # travel22: 2 aspects × 2 preferences each (K=4)
        "parquets":     ["data/travel22_multiturn_onechoice"],
        "start_idx":    0,
        "num_episodes": 50,
    },
    {
        "name":         "medium",
        # Restricted to travel33 only: 3 aspects × 3 preferences each (K=9).
        # travel233 (2 aspects, 3 prefs) dropped — difficulty axis is purely
        # preferences per aspect: easy=2, medium=3, hard=4.
        "parquets":     [
            "data/travel33_multiturn_onechoice",
        ],
        "start_idx":    0,
        "num_episodes": 50,
    },
    {
        "name":         "hard",
        # Restricted to travel444 only: 4 aspects × 4 preferences each (K=16).
        # travel334 dropped for same reason as travel233 above.
        "parquets":     [
            "data/travel444_multiturn_onechoice",
        ],
        "start_idx":    0,
        "num_episodes": 50,
    },
]

JSON_DIR = Path("travelgym/data")
RAW_DIR  = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_TO_DIM = {
    "F": "flight",
    "A": "apartment",
    "C": "rental_car",
    "H": "hotel",
    "R": "restaurant",
}

# ── system prompt addition ────────────────────────────────────────────────────
ANSWER_REMINDER = """
CRITICAL RULE FOR [answer] ACTION:
When you use [answer], the content field must contain ONLY the option ID.
Nothing else. No description, no explanation, no question.
Examples of correct [answer] usage:
  [answer] C16
  [answer] A17
  [answer] F3
The option ID is always a letter followed by a number (e.g. C16, A17, F3, H5, R2).
You can see the IDs in the search results next to each option.
WRONG: [answer] I recommend the Toyota Corolla compact car for $250
RIGHT: [answer] C16
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helper: parse structured thought JSON
# ─────────────────────────────────────────────────────────────────────────────

def parse_thought(thought_str: str) -> dict:
    """
    Attempts to parse the agent's thought field as structured JSON.

    Expected format:
        {
          "inferred_so_far": ["pref1", "pref2", ...],
          "still_unclear":   ["pref3", ...],
          "action_reason":   "one sentence"
        }

    Returns a dict with those three keys. On parse failure (agent wrote prose
    or malformed JSON), falls back gracefully: raw thought is preserved in
    "action_reason" and the lists are empty, so downstream code never crashes.

    The "parse_ok" flag lets us track compliance rate across episodes.
    """
    import re as _re

    # Strip markdown code fences if the agent wrapped JSON in ```json ... ```
    cleaned = thought_str.strip()
    fence_match = _re.match(r"^```(?:json)?\s*([\s\S]*?)```$", cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        parsed = json.loads(cleaned)
        inferred = parsed.get("inferred_so_far", [])
        still    = parsed.get("still_unclear", [])
        reason   = parsed.get("action_reason", "")
        if isinstance(inferred, list) and isinstance(still, list) and isinstance(reason, str):
            return {
                "inferred_so_far": inferred,
                "still_unclear":   still,
                "action_reason":   reason,
                "parse_ok":        True,
                "raw":             thought_str,
            }
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: return empty structure, preserve raw text
    return {
        "inferred_so_far": [],
        "still_unclear":   [],
        "action_reason":   thought_str,
        "parse_ok":        False,
        "raw":             thought_str,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load and pool parquets for a tier
# ─────────────────────────────────────────────────────────────────────────────

def load_tier_df(parquet_dirs: list) -> pd.DataFrame:
    frames = []
    for d in parquet_dirs:
        p = Path(d) / "test.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing parquet: {p}")
        df = pd.read_parquet(p)
        print(f"  Loaded {len(df)} rows from {p}")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Pooled + shuffled: {len(combined)} rows total")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load episode config
# ─────────────────────────────────────────────────────────────────────────────

def load_episode_config(df: pd.DataFrame, row_index: int) -> dict:
    print(f"  Loading row {row_index} from dataframe")
    row = df.iloc[row_index]

    reward_model    = row["reward_model"]
    scenario_id     = reward_model["id"]
    prompt          = list(row["prompt"])
    system_prompt   = prompt[0]["content"]
    initial_request = prompt[1]["content"]

    print(f"  scenario_id = {scenario_id}")

    for jfile in sorted(JSON_DIR.glob("travelgym_data_*.json")):
        with open(jfile) as f:
            raw = json.load(f)
        if scenario_id not in raw:
            continue

        scenario = raw[scenario_id]
        dims     = scenario["dimensions"]

        u_star               = {}
        implicit_expressions = {}
        best_ids             = {}
        correct_ids          = {}

        for dim in dims:
            prefs = scenario[dim]["preferences"]
            u_star[dim]               = [p[2] for p in prefs]
            implicit_expressions[dim] = [p[3] for p in prefs]
            best_ids[dim]             = scenario[dim]["best_id"]
            correct_ids[dim]          = scenario[dim]["correct_ids"]

        print(f"  JSON file   = {jfile.name}")
        print(f"  dimensions  = {dims}")
        print(f"  difficulty  = {scenario.get('difficulty', '?')}")
        print(f"  best_ids    = {best_ids}")
        print(f"  correct_ids = {correct_ids}")

        return {
            "scenario_id":          scenario_id,
            "system_prompt":        system_prompt,
            "initial_request":      initial_request,
            "dimensions":           dims,
            "u_star":               u_star,
            "implicit_expressions": implicit_expressions,
            "best_ids":             best_ids,
            "correct_ids":          correct_ids,
        }

    raise ValueError(
        f"scenario_id '{scenario_id}' not found in any JSON under {JSON_DIR}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper: single agent turn
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_turn(api_messages: list):
    response = client.chat.completions.create(
        model=AGENT_MODEL,
        messages=api_messages,
        tools=[TOOL_SCHEMA],
        tool_choice="required",
        temperature=0.7,
        max_completion_tokens=1024,
    )

    msg       = response.choices[0].message
    tool_call = msg.tool_calls[0]
    args      = json.loads(tool_call.function.arguments)

    thought = args.get("thought", {})  # dict with new schema, str with old
    choice  = args.get("choice", "")
    content = args.get("content", "")

    return thought, choice, content, msg, tool_call


# ─────────────────────────────────────────────────────────────────────────────
# Helper: per-dimension action scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_answers(answers_given: dict, config: dict) -> dict:
    """
    Returns per-dimension action scores and summary.

    action_score per dim:
        1.0  if chosen == best
        0.8  if chosen in correct (but not best)
        0.0  otherwise
    """
    dim_results   = {}
    all_dims_best = True
    any_correct   = False

    for dim in config["dimensions"]:
        init   = next(k for k, v in INITIAL_TO_DIM.items() if v == dim)
        chosen = answers_given.get(init)
        best   = config["best_ids"][dim]
        corr   = config["correct_ids"][dim]

        is_best    = chosen == best
        is_correct = chosen in corr if chosen else False

        if is_best:
            action_score = 1.0
        elif is_correct:
            action_score = 0.8
        else:
            action_score = 0.0

        if not is_best:
            all_dims_best = False
        if is_correct or is_best:
            any_correct = True

        dim_results[dim] = {
            "chosen":       chosen,
            "best":         best,
            "correct_ids":  corr,
            "action_score": action_score,
            "is_best":      is_best,
            "is_correct":   is_correct,
        }

        flag = "BEST" if is_best else ("CORRECT" if is_correct else "WRONG/MISSING")
        print(f"  {dim:12s}  chosen={chosen}  best={best}  "
              f"action_score={action_score}  → {flag}")

    return {
        "dim_results":      dim_results,
        "did_pick_best":    all_dims_best,
        "did_pick_correct": any_correct,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main: run one episode
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(tier: str, config: dict, max_turns: int = 20) -> dict:
    print(f"\n{'='*60}")
    print(f"  {tier.upper()} EPISODE")
    print(f"  Scenario ID : {config['scenario_id']}")
    print(f"  Dimensions  : {config['dimensions']}")
    print(f"{'='*60}")

    import travelgym

    env_config                       = travelgym.get_default_config()
    env_config.max_steps             = max_turns
    env_config.data_mode             = "single"
    env_config.data_source           = config["scenario_id"]
    env_config.one_choice_per_aspect = True
    env_config.wrong_choice_number   = 10
    env_config.noise_choice_number   = 5
    env_config.verbose               = False

    env = travelgym.TravelEnv(config=env_config)
    obs, info = env.reset()
    print(f"  Env ready.  task_id={info['task_id']}")
    print(f"  Total preferences to elicit: {obs['total_preferences']}")

    enhanced_system_prompt = config["system_prompt"] + ANSWER_REMINDER

    api_messages = [
        {"role": "system", "content": enhanced_system_prompt},
        {"role": "user",   "content": config["initial_request"]},
    ]
    clean_history = [
        {"role": "system", "content": config["system_prompt"]},  # save original
        {"role": "user",   "content": config["initial_request"]},
    ]

    # ── tracking ──────────────────────────────────────────────────────────────
    answers_given        = {}   # dim_initial → option_id
    answer_thoughts      = {}   # dim_initial → thought at answer time
    thought_trace        = []   # [{turn, choice, thought}, ...]
    turn_count           = 0
    terminated_naturally = False

    # ── turn loop ─────────────────────────────────────────────────────────────
    for turn in range(max_turns):
        turn_count = turn + 1
        print(f"\n  ── Turn {turn_count} ──")

        try:
            thought, choice, content, assistant_msg, tool_call = run_agent_turn(
                api_messages
            )
        except Exception as e:
            print(f"  AGENT CALL FAILED at turn {turn_count}: {e}")
            traceback.print_exc()
            break

        formatted_action = f"[{choice}] {content}"
        print(f"  [AGENT | {choice.upper()}] {formatted_action[:200]}")

        # thought is now a dict from the structured tool schema
        if isinstance(thought, dict):
            parsed_thought = {
                "inferred_so_far": thought.get("inferred_so_far", []),
                "still_unclear":   thought.get("still_unclear", []),
                "action_reason":   thought.get("action_reason", ""),
                "parse_ok":        True,
                "raw":             thought,
            }
        else:
            # Fallback: model returned a string (shouldn't happen with new schema)
            parsed_thought = parse_thought(thought)

        print(f"  [INFERRED] {parsed_thought['inferred_so_far']}")
        print(f"  [UNCLEAR]  {parsed_thought['still_unclear']}")
        print(f"  [REASON]   {parsed_thought['action_reason']}")
        if not parsed_thought["parse_ok"]:
            print(f"  [WARN] thought not valid JSON — parse_ok=False")

        # save thought to trace
        thought_trace.append({
            "turn":           turn_count,
            "choice":         choice,
            "thought":        thought,
            "parsed_thought": parsed_thought,
        })

        # save to clean_history
        clean_history.append({
            "role":           "agent",
            "content":        formatted_action,
            "thought":        thought,
            "parsed_thought": parsed_thought,
        })

        try:
            obs, reward, terminated, truncated, info = env.step(formatted_action)
        except Exception as e:
            print(f"  ERROR in env.step() at turn {turn_count}: {e}")
            traceback.print_exc()
            raise

        feedback = obs["feedback"]
        env_role = "database" if choice == "search" else "user"

        feedback_preview = (
            feedback[:250] + f"\n  ...(truncated)"
            if len(feedback) > 400 else feedback
        )
        print(f"  [{env_role.upper()}] {feedback_preview}")
        print(f"  reward={reward:.3f}  terminated={terminated}  truncated={truncated}")

        clean_history.append({"role": env_role, "content": feedback})

        # track answers + answer-time thoughts
        if choice == "answer":
            option_id = content.strip()
            if option_id and option_id[0] in INITIAL_TO_DIM:
                dim_initial                  = option_id[0]
                answers_given[dim_initial]   = option_id
                answer_thoughts[dim_initial] = thought
                print(f"  → Recorded answer : {option_id} "
                      f"for {INITIAL_TO_DIM[dim_initial]}")
                print(f"  → Answer reason   : {parsed_thought['action_reason']}")

        # update api_messages
        api_feedback = (
            feedback[:256] + "  ... " + feedback[-256:]
            if choice == "search" and len(feedback) > 512
            else feedback
        )

        assistant_message = {
            "role":    "assistant",
            "content": assistant_msg.content,
            "tool_calls": [{
                "id":   tool_call.id,
                "type": "function",
                "function": {
                    "name":      tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                }
            }]
        }
        api_messages.append(assistant_message)
        api_messages.append({
            "role":         "tool",
            "tool_call_id": tool_call.id,
            "content":      api_feedback if isinstance(api_feedback, str) else str(api_feedback),
        })

        if terminated or truncated:
            if terminated:
                terminated_naturally = True
            print(f"\n  Episode ended: terminated={terminated} truncated={truncated}")
            break

        time.sleep(1)

    # ── score results ─────────────────────────────────────────────────────────
    print(f"\n  RESULTS  (answers_given={answers_given})")

    # Thought parse compliance summary
    n_total   = len(thought_trace)
    n_ok      = sum(1 for t in thought_trace if t.get("parsed_thought", {}).get("parse_ok", False))
    print(f"  Thought parse compliance: {n_ok}/{n_total} turns had valid JSON thought")

    scoring = score_answers(answers_given, config)

    has_any_task_success = scoring["did_pick_correct"]
    print(f"\n  has_any_task_success : {has_any_task_success}")
    print(f"  did_pick_best        : {scoring['did_pick_best']}")

    return {
        "scenario_id":               config["scenario_id"],
        "tier":                      tier,
        "dimensions":                config["dimensions"],
        "u_star":                    config["u_star"],
        "implicit_expressions":      config["implicit_expressions"],
        "best_ids":                  config["best_ids"],
        "correct_ids":               config["correct_ids"],
        "answers_given":             answers_given,           # {dim_initial: option_id}
        "answer_thoughts":           answer_thoughts,         # {dim_initial: thought at answer time}
        "thought_trace":             thought_trace,           # [{turn, choice, thought, parsed_thought}, ...]
        "dim_results":               scoring["dim_results"],  # per-dim action scores
        "conversation_history":      clean_history,           # includes thought + parsed_thought per agent turn
        "turn_count":                turn_count,
        "terminated_naturally":      terminated_naturally,
        "did_pick_best":             scoring["did_pick_best"],
        "did_pick_correct":          scoring["did_pick_correct"],
        "has_any_task_success":      has_any_task_success,
        "agent_model":               AGENT_MODEL,
        "thought_parse_compliance":  {"ok": n_ok, "total": n_total},  # fraction of turns with valid JSON thought
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'#'*60}")
    print("  generate_samples.py — TravelEnv multi-tier episodes")
    print(f"{'#'*60}\n")

    for tier_cfg in TIER_CONFIGS:
        tier         = tier_cfg["name"]
        start_idx    = tier_cfg["start_idx"]
        num_episodes = tier_cfg["num_episodes"]

        if num_episodes == 0:
            print(f"\nSkipping {tier} (num_episodes=0)")
            continue

        print(f"\n{'#'*60}")
        print(f"  TIER: {tier.upper()}  start_idx={start_idx}  "
              f"num_episodes={num_episodes}")
        print(f"{'#'*60}")

        print(f"\n[STEP 1] Loading parquets for {tier}...")
        df = load_tier_df(tier_cfg["parquets"])

        succeeded = 0
        skipped   = 0

        for i in range(num_episodes):
            row_index  = start_idx + i
            global_idx = start_idx + i
            save_path  = str(RAW_DIR / f"{tier}_{global_idx:03d}.json")

            print(f"\n{'#'*60}")
            print(f"  [{tier.upper()}] Episode {i+1}/{num_episodes}  "
                  f"row={row_index}  → {save_path}")
            print(f"{'#'*60}")

            try:
                config = load_episode_config(df, row_index=row_index)
                result = run_episode(tier, config, max_turns=20)

                with open(save_path, "w") as f:
                    json.dump(result, f, indent=2)

                print(f"\n  Saved → {save_path}")
                print(f"  has_any_task_success={result['has_any_task_success']}  "
                      f"did_pick_best={result['did_pick_best']}")
                succeeded += 1

            except Exception as e:
                print(f"\n  WARNING: {tier} episode {i+1} "
                      f"(row={row_index}) FAILED — skipping.")
                print(f"  {type(e).__name__}: {e}")
                traceback.print_exc()
                skipped += 1

            if i < num_episodes - 1:
                time.sleep(2)

        print(f"\n  [{tier.upper()}] Done: {succeeded} succeeded, {skipped} failed.")
        print(f"  Files: {RAW_DIR}/{tier}_{start_idx:03d}.json … "
              f"{tier}_{start_idx+num_episodes-1:03d}.json")

    print(f"\n\n{'#'*60}")
    print("  All tiers complete.")
    print(f"{'#'*60}\n")