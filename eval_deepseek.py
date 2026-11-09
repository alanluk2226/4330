"""Correct saved LoRA drafts with DeepSeek and score MER.

References are used only to compute MER. They are not sent to the API.
Thinking mode is disabled so each call returns one short line.
"""

import argparse
import json
import os
import urllib.error
import urllib.request

from correct_sample import SYSTEM_INSTRUCTION, load_env
from mce_common import RESULTS_DIR, mixed_error_rate
from score_no_simplified import has_simplified

MODEL_NAME = "deepseek-flash"
API_URL = "https://api.deepseek.com/chat/completions"
PAIRS_PATH = RESULTS_DIR / "lora_prompt_pairs.jsonl"
GUARD_INSTRUCTION = """
You are a specialized post-processing engine for Cantonese-English code-switching ASR transcripts.

The input is one Whisper + LoRA draft. It may contain a few homophone or spelling errors. Your ONLY task is to fix obvious acoustic or phonetic errors.

STRICT CONSTRAINTS:
1. Preserve English terms and proper nouns. Correct an English word only when it is an obvious mishearing or misspelling of a known word, for example "Fan is" or "Fun is" for Venice, or a phonetic rendering of "gondola". Keep English in English. Do not translate it, and do not replace it with an arbitrary word.
2. Keep colloquial Cantonese. Do not rewrite particles, grammar, or tone into written Chinese. Keep 咁、嘅、囉、係、真係、我哋、咗.
3. Zero lexical drift. Do not replace a natural word with a synonym. Do not swap 使你 and 似你.
4. Repair a fixed Cantonese expression only when a homophone breaks it, such as 大意滅親 back to 大義滅親. In a dim-sum or table-booking context, treat 早餐 as 早茶.
5. No over-correction. If a sentence is already accurate and natural, output it exactly as received.
6. Output ONLY the corrected text. No explanations and no markdown.
"""


def load_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def first_line(text):
    for line in text.strip().splitlines():
        line = line.strip().strip("\"'")
        if line:
            return line
    return ""


def correct(api_key, draft, instruction, temperature):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": draft},
        ],
        "temperature": temperature,
        "max_tokens": 192,
        "thinking": {"type": "disabled"},
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise SystemExit(f"DeepSeek HTTP {error.code}: {detail[:500]}") from error
    message = body["choices"][0]["message"]
    return message.get("content") or "", body.get("usage", {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--prompt", choices=("gemini", "guard"), default="gemini")
    parser.add_argument("--no-simplified", action="store_true")
    args = parser.parse_args()
    load_env()
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is missing from .env")

    pairs = load_jsonl(PAIRS_PATH)
    if len(pairs) != 1405:
        raise SystemExit(f"Expected 1405 LoRA drafts, found {len(pairs)}")
    if args.no_simplified:
        pairs = [pair for pair in pairs if not has_simplified(pair["reference"])]
    if args.limit:
        pairs = pairs[: args.limit]

    instruction = SYSTEM_INSTRUCTION if args.prompt == "gemini" else GUARD_INSTRUCTION
    temperature = 1 if args.prompt == "gemini" else 0
    suffix = "" if args.prompt == "gemini" else "_guard"
    if args.no_simplified:
        suffix += "_nosimp"
    progress_path = RESULTS_DIR / f"lora_deepseek{suffix}_progress.jsonl"
    output_path = RESULTS_DIR / f"lora_deepseek{suffix}.json"
    done = {row["index"]: row for row in load_jsonl(progress_path)}
    print(f"DeepSeek progress {len(done)}/{len(pairs)}", flush=True)

    for pair in pairs:
        if pair["index"] in done:
            continue
        raw, usage = correct(api_key, pair["hypothesis"], instruction, temperature)
        row = {
            "index": pair["index"],
            "reference": pair["reference"],
            "hypothesis": pair["hypothesis"],
            "deepseek": first_line(raw),
            "deepseek_raw": raw.strip(),
            "usage": usage,
            "model": MODEL_NAME,
        }
        append_jsonl(progress_path, row)
        done[pair["index"]] = row
        if args.limit and args.limit <= 30:
            print(f"[{pair['index']}]")
            print(f"Reference: {pair['reference']}")
            print(f"LoRA:      {pair['hypothesis']}")
            print(f"DeepSeek:  {row['deepseek']}")
            print(flush=True)
        elif len(done) % 50 == 0:
            print(f"DeepSeek progress {len(done)}/{len(pairs)}", flush=True)

    ordered = [done[pair["index"]] for pair in pairs]
    deepseek_mer = mixed_error_rate((row["reference"], row["deepseek"]) for row in ordered)
    lora_mer = mixed_error_rate((row["reference"], row["hypothesis"]) for row in ordered)
    qwen_rows = {row["index"]: row for row in load_jsonl(RESULTS_DIR / "lora_qwen_progress.jsonl")}
    qwen_pairs = [
        (row["reference"], qwen_rows[row["index"]]["qwen"])
        for row in ordered
        if row["index"] in qwen_rows
    ]
    qwen_mer = mixed_error_rate(qwen_pairs) if len(qwen_pairs) == len(ordered) else None
    payload = {
        "model": MODEL_NAME,
        "thinking": "disabled",
        "temperature": temperature,
        "prompt": args.prompt,
        "sentences": len(ordered),
        "lora_mer": lora_mer,
        "qwen_mer": qwen_mer,
        "deepseek_mer": deepseek_mer,
        "note": "Sample of the saved LoRA drafts. References were not sent to DeepSeek.",
        "examples": [
            {
                "reference": row["reference"],
                "lora": row["hypothesis"],
                "deepseek": row["deepseek"],
            }
            for row in ordered[:20]
        ],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Sentences    = {len(ordered)}")
    print(f"LoRA MER     = {lora_mer:.2%}")
    if qwen_mer is not None:
        print(f"Qwen MER     = {qwen_mer:.2%}")
    print(f"DeepSeek MER = {deepseek_mer:.2%}")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
