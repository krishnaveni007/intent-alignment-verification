import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any
from openai import OpenAI

# dummy function to load environment var because I ran into an environment issue
# could change to use python-dotenv when I fix the issue
def load_env(file="../.env"):
    with open(file) as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                os.environ[key] = value

load_env()

SUPPORTED_MODELS = {"gpt-4o", "gpt-4o-mini"}

def load_api_key(file_path: str) -> str:
    with open(file_path, "r") as f:
        return f.read().strip()

def load_json_file(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# validate sample json file
def validate_input(data: Dict[str, Any]) -> None:
    if "conversation_history" not in data:
        raise ValueError("Input JSON must contain 'conversation_history'.")
    if "dimensions" not in data:
        raise ValueError("Input JSON must contain 'dimensions'.")
    if not isinstance(data["conversation_history"], list):
        raise ValueError("'conversation_history' must be a list.")
    if not isinstance(data["dimensions"], list):
        raise ValueError("'dimensions' must be a list.")

# only gpt-4 does not support schema, gpt-4o and gpt-4o-mini do
def build_schema(dimensions: List[str]) -> Dict[str, Any]:
    return {
        "name": "user_preferences_by_category",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                dim: {
                    "type": "string",
                    "description": (
                        f"Inferred user preference for category '{dim}'. "
                    ),
                }
                for dim in dimensions
            },
            "required": dimensions,
            "additionalProperties": False,
        },
    }

# build task description, input to LLM
def build_messages(conversation_history: List[Dict[str, str]], dimensions: List[str]) -> List[Dict[str, str]]:
    system_instruction = {
        "role": "system",
        "content": (
            "You are an expert at inferring user preferences from dialogue.\n\n"
            "Task:\n"
            "Given the conversation history, infer the user's preferences for each category.\n\n"
            "Rules:\n"
            f"1. Only infer preferences for these categories: {', '.join(dimensions)}.\n"
            "2. Output must match the required JSON schema exactly.\n"
            "3. Each category value should summarize the user's inferred preference.\n"
            "4. Base your answer only on evidence from the conversation.\n"
            "5. Do not include explanations or extra keys."
        ),
    }

    user_prompt = {
        "role": "user",
        "content": (
            "Infer the user's preferences from the following conversation history.\n\n"
            f"Target categories: {json.dumps(dimensions, ensure_ascii=False)}\n\n"
            "Conversation history:\n"
            f"{json.dumps(conversation_history, ensure_ascii=False, indent=2)}"
        ),
    }

    return [system_instruction, user_prompt]

# main inference function
def infer_preferences(file_path: str, model: str) -> Dict[str, str]:
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model}. Supported models: {SUPPORTED_MODELS}")

    data = load_json_file(file_path)
    validate_input(data)

    conversation_history = data["conversation_history"]
    dimensions = data["dimensions"]

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment variables.")

    client = OpenAI(api_key=api_key)

    schema = build_schema(dimensions)
    messages = build_messages(conversation_history, dimensions)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": schema,
        },
        temperature=0,
    )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Model returned empty content.")

    return json.loads(content)

def main():
    if len(sys.argv) != 3:
        print("Usage: python llm_infer.py <json_file> <model>")
        sys.exit(1)

    json_file = sys.argv[1]
    model = sys.argv[2]

    if not Path(json_file).exists():
        print(f"File not found: {json_file}")
        sys.exit(1)

    result = infer_preferences(json_file, model)

    print(json.dumps(result, ensure_ascii=False, indent=2))

    output_file = Path(json_file).stem + "_out.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Output written to {output_file}")

if __name__ == "__main__":
    main()