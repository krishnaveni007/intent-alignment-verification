#!/usr/bin/env python3
"""
generate_samples.py

Run 3 real TravelEnv episodes (Easy / Medium / Hard) using gpt-4o as
the RL agent and GPT-4o (env default) as the user/search simulator.

Usage:
    export OPENAI_API_KEY="sk-..."
    python3 generate_samples.py

Outputs:
    results/sample_easy.json
    results/sample_medium.json
    results/sample_hard.json
"""

import os
import sys
import json
import time
import yaml
import traceback
from pathlib import Path

import pandas as pd

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

# ── tier configs ──────────────────────────────────────────────────────────────
TIERS = [
    ("easy",   "data/travel22_multiturn_onechoice/test.parquet"),
    ("medium", "data/travel33_multiturn_onechoice/test.parquet"),
    ("hard",   "data/travel334_multiturn_onechoice/test.parquet"),
]

JSON_DIR    = Path("travelgym/data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# initial letter → dimension name (used for evaluating answers)
INITIAL_TO_DIM = {
    "F": "flight",
    "A": "apartment",
    "C": "rental_car",
    "H": "hotel",
    "R": "restaurant",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load episode config
# ─────────────────────────────────────────────────────────────────────────────

def load_episode_config(parquet_path: str, row_index: int = 0) -> dict:
    """
    Load one parquet row and the matching JSON ground truth.
    Returns a dict with all fields needed to run the episode.
    """
    print(f"  Reading parquet: {parquet_path} (row {row_index})")
    df = pd.read_parquet(parquet_path)
    row = df.iloc[row_index]

    reward_model    = row["reward_model"]
    scenario_id     = reward_model["id"]
    prompt          = list(row["prompt"])
    system_prompt   = prompt[0]["content"]
    initial_request = prompt[1]["content"]

    print(f"  scenario_id = {scenario_id}")

    # Search JSON files for this scenario
    for jfile in sorted(JSON_DIR.glob("travelgym_data_*.json")):
        with open(jfile) as f:
            raw = json.load(f)
        if scenario_id not in raw:
            continue

        scenario = raw[scenario_id]
        dims     = scenario["dimensions"]

        u_star              = {}
        implicit_expressions = {}
        best_ids            = {}
        correct_ids         = {}

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
        print(f"  u_star:")
        for dim, prefs in u_star.items():
            print(f"    {dim}: {prefs}")
        print(f"  implicit_expressions:")
        for dim, exprs in implicit_expressions.items():
            for e in exprs:
                print(f"    [{dim}] {e[:80]}...")

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
# Helper: single agent turn (tool call)
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_turn(api_messages: list):
    """
    Call gpt-4o with the interact_with_env tool.
    Returns (thought, choice, content, assistant_message, tool_call).
    Raises on failure (do not silently swallow errors).
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=api_messages,
        tools=[TOOL_SCHEMA],
        tool_choice="required",
        temperature=0.7,
        max_tokens=1024,
    )

    msg       = response.choices[0].message
    tool_call = msg.tool_calls[0]
    args      = json.loads(tool_call.function.arguments)

    thought = args.get("thought", "")
    choice  = args.get("choice", "")
    content = args.get("content", "")

    return thought, choice, content, msg, tool_call


# ─────────────────────────────────────────────────────────────────────────────
# Main: run one episode
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(tier: str, config: dict, max_turns: int = 10) -> dict:
    """Run a full TravelEnv episode and return the result dict."""

    print(f"\n{'='*60}")
    print(f"  {tier.upper()} EPISODE")
    print(f"  Scenario ID : {config['scenario_id']}")
    print(f"  Dimensions  : {config['dimensions']}")
    print(f"  Max turns   : {max_turns}")
    print(f"{'='*60}")

    # ── initialise env ────────────────────────────────────────────────────────
    import travelgym

    env_config                    = travelgym.get_default_config()
    env_config.max_steps          = max_turns
    env_config.data_mode          = "single"
    env_config.data_source        = config["scenario_id"]
    env_config.one_choice_per_aspect = True
    env_config.wrong_choice_number   = 10
    env_config.noise_choice_number   = 5
    env_config.verbose               = False  # we do our own printing

    print("  Initialising TravelEnv (loads all JSON, finds scenario)...")
    env = travelgym.TravelEnv(config=env_config)
    obs, info = env.reset()
    print(f"  Env ready.  task_id={info['task_id']}")
    print(f"  Total preferences to elicit: {obs['total_preferences']}")

    # ── initial message lists ─────────────────────────────────────────────────
    # api_messages  → sent to the LLM agent each turn
    # clean_history → saved in the output JSON
    answer_reminder = """
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
    enhanced_system_prompt = config["system_prompt"] + answer_reminder

    api_messages = [
        {"role": "system", "content": enhanced_system_prompt},
        {"role": "user",   "content": config["initial_request"]},
    ]
    clean_history = [
        {"role": "system", "content": config["system_prompt"]},  # save original, not enhanced
        {"role": "user",   "content": config["initial_request"]},
    ]

    # ── tracking ──────────────────────────────────────────────────────────────
    answers_given       = {}   # initial_letter → option_id  e.g. {"F": "F3", "A": "A1"}
    turn_count          = 0
    terminated_naturally = False

    # ── turn loop ─────────────────────────────────────────────────────────────
    for turn in range(max_turns):
        turn_count = turn + 1
        print(f"\n  ── Turn {turn_count} ──")

        # -- agent turn --------------------------------------------------------
        try:
            thought, choice, content, assistant_msg, tool_call = run_agent_turn(
                api_messages
            )
        except Exception as e:
            print(f"  AGENT CALL FAILED at turn {turn_count}")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Error message: {e}")
            traceback.print_exc()
            break

        formatted_action = f"[{choice}] {content}"

        print(f"  [AGENT | {choice.upper()}]")
        print(f"    thought : {thought[:120]}")
        print(f"    action  : {formatted_action[:200]}")

        clean_history.append({"role": "agent", "content": formatted_action})

        # -- env step ----------------------------------------------------------
        try:
            obs, reward, terminated, truncated, info = env.step(formatted_action)
        except Exception as e:
            print(f"\n  ERROR in env.step() at turn {turn_count}!")
            print(f"  Agent response that caused it: {formatted_action}")
            traceback.print_exc()
            raise   # stop execution, full traceback visible

        feedback = obs["feedback"]
        env_role = "database" if choice == "search" else "user"

        # Trim long search results for terminal readability
        if len(feedback) > 400:
            feedback_preview = feedback[:250] + f"\n  ... (total {len(feedback)} chars, truncated) ..."
        else:
            feedback_preview = feedback

        print(f"  [{env_role.upper()}]")
        print(f"    {feedback_preview}")
        print(f"    reward={reward:.3f}  terminated={terminated}  truncated={truncated}")
        print(f"    elicited {obs.get('total_preferences',0)-obs.get('remaining_preferences',0)} / {obs.get('total_preferences',0)} prefs so far")

        clean_history.append({"role": env_role, "content": feedback})

        # track answer choices
        if choice == "answer":
            option_id = content.strip()
            if option_id and option_id[0] in INITIAL_TO_DIM:
                answers_given[option_id[0]] = option_id
                print(f"    → Recorded answer: {option_id} for {INITIAL_TO_DIM[option_id[0]]}")

        # -- update api_messages (tool call → tool result) --------------------
        # Trim feedback in api_messages to avoid enormous context for search
        if choice == "search" and len(feedback) > 512:
            api_feedback = feedback[:256] + "  ... ... " + feedback[-256:]
        else:
            api_feedback = feedback

        # Build assistant message with tool call
        assistant_message = {
            "role": "assistant",
            "content": assistant_msg.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    }
                }
            ]
        }
        api_messages.append(assistant_message)

        # Tool result message — id MUST match the one above exactly
        tool_result_message = {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": api_feedback if isinstance(api_feedback, str) else str(api_feedback),
        }
        api_messages.append(tool_result_message)

        # Verify ids match before next turn — crash loudly if not
        assert assistant_message["tool_calls"][0]["id"] == tool_result_message["tool_call_id"], \
            f"MISMATCH: {assistant_message['tool_calls'][0]['id']} != {tool_result_message['tool_call_id']}"

        if terminated or truncated:
            if terminated:
                terminated_naturally = True
            print(f"\n  Episode ended: terminated={terminated} truncated={truncated}")
            break

        # small pause to respect rate limits
        time.sleep(1)

    # ── evaluate results ──────────────────────────────────────────────────────
    all_dims_best    = True
    any_dim_correct  = False

    print(f"\n  {'─'*40}")
    print(f"  RESULTS")
    print(f"  Answers given : {answers_given}")
    for dim in config["dimensions"]:
        init = next(k for k, v in INITIAL_TO_DIM.items() if v == dim)
        chosen = answers_given.get(init)
        best   = config["best_ids"][dim]
        corr   = config["correct_ids"][dim]

        is_best    = chosen == best
        is_correct = chosen in corr if chosen else False

        if not is_best:
            all_dims_best = False
        if is_correct:
            any_dim_correct = True

        flag = "BEST" if is_best else ("CORRECT" if is_correct else "WRONG/MISSING")
        print(f"  {dim:12s}  chosen={chosen}  best={best}  → {flag}")

    print(f"  did_pick_best    : {all_dims_best}")
    print(f"  did_pick_correct : {any_dim_correct}")

    return {
        "scenario_id":          config["scenario_id"],
        "tier":                 tier,
        "dimensions":           config["dimensions"],
        "u_star":               config["u_star"],
        "implicit_expressions": config["implicit_expressions"],
        "best_ids":             config["best_ids"],
        "correct_ids":          config["correct_ids"],
        "conversation_history": clean_history,
        "turn_count":           turn_count,
        "terminated_naturally": terminated_naturally,
        "did_pick_best":        all_dims_best,
        "did_pick_correct":     any_dim_correct,
        "agent_model":          "gpt-4o",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def pretty_print_result(result: dict, save_path: str):
    """Print a readable episode summary to the terminal."""
    tier = result["tier"].upper()
    SEP  = "─" * 50
    print(f"\n{'='*50}")
    print(f"=== {tier} TIER SAMPLE ===")
    print(f"Scenario ID: {result['scenario_id']}")
    print(f"Dimensions : {', '.join(result['dimensions'])}")
    print("u* (hidden ground truth):")
    for dim, prefs in result["u_star"].items():
        print(f"  {dim}: {prefs}")
    print(SEP)

    agent_turn_num = 0
    for i, entry in enumerate(result["conversation_history"]):
        role    = entry["role"]
        content = entry["content"]

        if role == "system":
            continue  # skip system prompt in pretty view

        if i == 1:  # initial user message
            print(f"[Initial Request - USER]:  {content[:150]}...")
            continue

        if role == "agent":
            agent_turn_num += 1
            # detect action type
            if content.startswith("[action]"):
                label = "ACTION"
            elif content.startswith("[search]"):
                label = "SEARCH"
            elif content.startswith("[answer]"):
                label = "ANSWER"
            else:
                label = "AGENT"
            disp = content[:200] + ("..." if len(content) > 200 else "")
            print(f"\n[Turn {agent_turn_num} - AGENT / {label}]:  {disp}")

        elif role == "user":
            disp = content[:200] + ("..." if len(content) > 200 else "")
            print(f"[Turn {agent_turn_num} - USER]:    {disp}")

        elif role == "database":
            disp = content[:150] + ("..." if len(content) > 150 else "")
            print(f"[Turn {agent_turn_num} - DATABASE]: {disp}")

    print(SEP)
    best_str    = "YES" if result["did_pick_best"]    else "NO"
    correct_str = "YES" if result["did_pick_correct"] else "NO"
    print(f"Turns : {result['turn_count']}")
    print(f"Result: picked best? {best_str} | picked correct? {correct_str}")
    print(f"Saved to {save_path}")
    print("=" * 50)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'#'*60}")
    print("  generate_samples.py — TravelEnv sample episodes")
    print(f"{'#'*60}\n")

    for tier, parquet_path in TIERS:
        print(f"\n{'#'*60}")
        print(f"  TIER: {tier.upper()}")
        print(f"{'#'*60}")

        print("\n[STEP 2] Loading episode config...")
        config = load_episode_config(parquet_path, row_index=0)

        print("\n[STEP 3/4] Running episode...")
        result = run_episode(tier, config, max_turns=20)

        save_path = str(RESULTS_DIR / f"sample_{tier}.json")
        print(f"\n[STEP 5] Saving result to {save_path}")
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2)

        print("\n[STEP 6] Pretty print:")
        pretty_print_result(result, save_path)

        # brief pause between episodes (not after the last one)
        if tier != "hard":
            print("\n[Pausing 3s before next episode...]")
            time.sleep(3)

    print(f"\n\n{'#'*60}")
    print("  All 3 episodes complete!")
    print(f"  Results saved to:")
    for tier, _ in TIERS:
        print(f"    {RESULTS_DIR}/sample_{tier}.json")
    print(f"{'#'*60}\n")
